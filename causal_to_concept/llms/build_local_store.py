#!/usr/bin/env python3
"""
Build local intervention store from training data.
Creates keys.npy and diffs/L{l}_H{h}.npy files for contextual interventions.
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from einops import rearrange
import json

def load_data_like_validate_2fold(dataset_name):
    """Load data the same way as validate_2fold_toxic.py"""
    if dataset_name == "toxigen_vicuna":
        toxigen_path = f'/work/hdd/bcxt/yian3/toxic/features/{dataset_name}_texts.json'
        with open(toxigen_path) as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
    elif dataset_name == "hate_vicuna":
        hate_path = f'/work/hdd/bcxt/yian3/toxic/features/{dataset_name}_texts.json'
        with open(hate_path) as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
    else:
        # Fallback to CSV loading
        df = pd.read_csv(f'./TruthfulQA/{dataset_name}.csv')
    
    print(f"Loaded {len(df)} examples from {dataset_name}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def load_all_data(dataset_name):
    """Load all data pairs (toxic vs non-toxic) from the dataset."""
    # Load the full dataset
    df = load_data_like_validate_2fold(dataset_name)
    
    # Extract texts from the dataframe
    if 'toxic_text' in df.columns and 'non_toxic_text' in df.columns:
        toxic_texts = df['toxic_text'].dropna().tolist()
        nontoxic_texts = df['non_toxic_text'].dropna().tolist()
        print(f"Using toxic_text and non_toxic_text columns")
    else:
        raise ValueError(f"Could not find 'toxic_text'/'non_toxic_text' columns in {df.columns.tolist()}")
    
    # Ensure same length
    min_len = min(len(toxic_texts), len(nontoxic_texts))
    toxic_texts = toxic_texts[:min_len]
    nontoxic_texts = nontoxic_texts[:min_len]
    
    print(f"Final dataset size: {len(toxic_texts)} pairs")
    
    return toxic_texts, nontoxic_texts

def get_activations_for_texts(texts, model, tokenizer, layer, head, device='cuda'):
    """Extract activations for a specific (layer, head) from texts."""
    activations = []
    
    for text in tqdm(texts, desc=f"Extracting activations for L{layer}H{head}"):
        try:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Get activations from the specified layer and head
                hidden_states = outputs.hidden_states[layer]  # [batch, seq_len, hidden_dim]
                num_heads = model.config.num_attention_heads
                head_dim = hidden_states.shape[-1] // num_heads
                
                # Reshape to separate heads
                hidden_states = rearrange(hidden_states, 'b s (h d) -> b s h d', h=num_heads)
                
                # Get the last token's activation for the specified head
                head_activation = hidden_states[0, -1, head, :].cpu().numpy()  # [head_dim]
                activations.append(head_activation)
                
        except Exception as e:
            print(f"Error processing text: {e}")
            # Use zero vector as fallback
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            activations.append(np.zeros(head_dim))
    
    return np.array(activations)

def load_selected_heads(heads_path):
    """Load pre-selected heads from numpy file."""
    if not os.path.exists(heads_path):
        print(f"Selected heads file not found: {heads_path}")
        return None
    
    selected_heads = np.load(heads_path)
    print(f"Loaded {len(selected_heads)} selected heads from {heads_path}")
    print(f"Selected heads: {selected_heads}")
    return selected_heads

def build_local_store_for_data(toxic_texts, nontoxic_texts, model, tokenizer, output_dir, data_name="", selected_heads=None):
    """Build local store for a specific dataset with per-input difference vectors."""
    if not toxic_texts or not nontoxic_texts:
        print(f"No data provided for {data_name}, skipping...")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "diffs"), exist_ok=True)
    
    print(f"\n=== Building {data_name} store ===")
    print(f"Processing {len(toxic_texts)} examples")
    
    # Generate sentence embeddings for keys
    print("Generating sentence embeddings...")
    sentence_embedding = SentenceTransformer('all-MiniLM-L6-v2')
    keys = sentence_embedding.encode(toxic_texts)
    
    # Load selected heads if provided
    if selected_heads is not None:
        print(f"Using pre-selected heads: {len(selected_heads)} heads")
        # Convert numpy int64 to Python int for JSON serialization
        head_combinations = [(int(layer), int(head)) for layer, head in selected_heads]
    else:
        # Fallback: process all heads (slow)
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        head_combinations = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]
        print(f"Processing all {num_layers} layers x {num_heads} heads = {len(head_combinations)} combinations")
    
    print(f"Processing {len(head_combinations)} (layer, head) combinations")
    
    for layer, head in head_combinations:
        # Check if diff file already exists
        diff_path = os.path.join(output_dir, "diffs", f"L{layer}_H{head}.npy")
        metadata_path = os.path.join(output_dir, "diffs", f"L{layer}_H{head}_metadata.json")
        
        if os.path.exists(diff_path) and os.path.exists(metadata_path):
            print(f"\nSkipping L{layer}H{head} - files already exist")
            print(f"  - {diff_path}")
            print(f"  - {metadata_path}")
            continue
            
        print(f"\nProcessing L{layer}H{head}...")
        
        # Get activations for toxic texts
        toxic_activations = get_activations_for_texts(
            toxic_texts, model, tokenizer, layer, head, device=model.device
        )
        
        # Get activations for non-toxic texts
        nontoxic_activations = get_activations_for_texts(
            nontoxic_texts, model, tokenizer, layer, head, device=model.device
        )
        
        # Compute per-input difference vectors
        diffs = nontoxic_activations - toxic_activations
        
        # Save diffs with metadata
        np.save(diff_path, diffs)
        
        # Save metadata for this head
        metadata = {
            'layer': layer,
            'head': head,
            'shape': diffs.shape,
            'num_examples': len(diffs),
            'head_dim': diffs.shape[1] if len(diffs.shape) > 1 else diffs.shape[0]
        }
        metadata_path = os.path.join(output_dir, "diffs", f"L{layer}_H{head}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved diffs to {diff_path}, shape: {diffs.shape}")
        print(f"  - Each row is a per-input difference vector (nontoxic - toxic)")
        print(f"  - Can be indexed by example index for specific interventions")
    
    # Save overall metadata
    overall_metadata = {
        'dataset_name': data_name,
        'num_examples': len(toxic_texts),
        'num_heads': len(head_combinations),
        'head_combinations': head_combinations,  # Already a list of Python tuples
        'description': 'Per-input difference vectors for contextual interventions'
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(overall_metadata, f, indent=2)
    
    print(f"\n{data_name} store built successfully in {output_dir}")
    print("Files created:")
    print(f"  - diffs/L{{l}}_H{{h}}.npy: {len(head_combinations)} files (per-input difference vectors)")
    print(f"  - diffs/L{{l}}_H{{h}}_metadata.json: {len(head_combinations)} files (per-head metadata)")
    print(f"  - metadata.json: Overall store metadata")
    print("\nUsage:")
    print("  - Load diffs/L{l}_H{h}.npy to get difference vectors for specific head")
    print("  - Index by example number to get specific difference vector")
    print("  - Use for contextual interventions during inference")

def build_local_store(args):
    """Build the local intervention store."""
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model_path = args.model_path if args.model_path else f"meta-llama/Meta-Llama-3-8B" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()
    
    # Set pad_token if not already set
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load selected heads if provided
    selected_heads = None
    if args.heads_path:
        selected_heads = load_selected_heads(args.heads_path)
    
    # Load all data
    print("Loading all data...")
    toxic_texts, nontoxic_texts = load_all_data(args.dataset_name)
    
    # Limit examples if specified
    if args.max_examples and len(toxic_texts) > args.max_examples:
        print(f"Limiting to {args.max_examples} examples")
        toxic_texts = toxic_texts[:args.max_examples]
        nontoxic_texts = nontoxic_texts[:args.max_examples]
    
    # Determine if using PNS heads from the heads_path
    use_pns = "True" if "True" in args.heads_path else False
    pns_suffix = "pns" if use_pns else "no_pns"
    
    # Create specific output directory structure
    specific_output_dir = os.path.join(
        args.output_dir, 
        f"{args.model_name}_{args.dataset_name}_{pns_suffix}"
    )
    
    # Build store with all data
    build_local_store_for_data(toxic_texts, nontoxic_texts, model, tokenizer, specific_output_dir, "all_data", selected_heads)
    
    print(f"\n=== Store built successfully in {specific_output_dir} ===")

def main():
    parser = argparse.ArgumentParser(description="Build local intervention store")
    parser.add_argument("--dataset_name", type=str, default="toxigen_vicuna", 
                       help="Dataset name")
    parser.add_argument("--model_name", type=str, default="llama3_8B", 
                       help="Model name")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to model (if different from model_name)")
    parser.add_argument("--output_dir", type=str, default="./local_store",
                       help="Output directory for local store")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to process (for testing)")
    parser.add_argument("--heads_path", type=str, default=None,
                       help="Path to pre-selected heads numpy file (e.g., top_heads.npy)")
    
    args = parser.parse_args()
    
    if args.max_examples:
        print(f"Limiting to {args.max_examples} examples for testing")
    
    build_local_store(args)

if __name__ == "__main__":
    main()


"""
python build_local_store.py \
  --dataset_name toxigen_vicuna \
  --model_name llama3_8B \
  --model_path /path/to/your/model \
  --output_dir ./local_store_toxigen

# For testing (limit examples)
python build_local_store.py \
  --dataset_name toxigen_vicuna \
  --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 \
  --heads_path /work/hdd/bcxt/yian3/toxic/features/heads/True_llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5_toxigen_vicuna_seed_2_top_72_heads_fold_0.npy \
  --output_dir /work/hdd/bcxt/yian3/toxic/local_store_toxigen 
python consolidate_diffs.py        
"""