#!/usr/bin/env python3
"""
Compare activation changes between two different models.

This script:
1. Loads top heads and interventions
2. Loads two different models
3. Collects activations before and after intervention for both models
4. Compares representation changes between the two models
5. Plots comparison visualizations

Usage example:
    python plot_activations_compare.py --num_heads 18 --alpha 5.0 \
        --model_name_1 llama3_8B --model_name_2 llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 \
        --dataset_name toxigen_vicuna \
        --heads_path /work/hdd/bcxt/yian3/toxic/features/heads/True_llama3_8B_toxigen_vicuna_seed_2_top_72_heads_fold_0.npy \
        --output_dir ./activation_plots_ft --num_samples 50
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyvene as pv

# Add path for utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from get_activations.interveners import Collector, ITI_Intervener, wrapper
from plot_activations import (
    load_top_heads,
    load_interventions,
    ActivationCollector,
    InterventionCollector,
    collect_activations_with_intervention,
    compute_representation_change,
)

# Model name mappings
HF_NAMES = {
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'vicuna_13B': 'lmsys/vicuna-13b-v1.5',
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'mistral_7B': 'mistralai/Mistral-7B-v0.1',
    'qwen_7B': 'Qwen/Qwen2.5-7B-Instruct',
}


def compare_models(changes_1, changes_2, top_heads, output_dir, alpha, num_heads, 
                   model_name_1, model_name_2, dataset_name):
    """Compare representation changes between two models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename prefix
    filename_parts = []
    if model_name_1:
        model_name_1_clean = model_name_1.replace('/', '_').replace('-', '_')
        filename_parts.append(model_name_1_clean)
    if model_name_2:
        model_name_2_clean = model_name_2.replace('/', '_').replace('-', '_')
        filename_parts.append(model_name_2_clean)
    if dataset_name:
        dataset_name_clean = dataset_name.replace('/', '_').replace('-', '_')
        filename_parts.append(dataset_name_clean)
    
    if filename_parts:
        filename_prefix = "_".join(filename_parts) + "_"
    else:
        filename_prefix = ""
    
    # Convert top_heads to tuples if they're numpy arrays
    top_heads_tuples = []
    for h in top_heads:
        if isinstance(h, np.ndarray):
            top_heads_tuples.append(tuple(h.tolist()))
        elif isinstance(h, list):
            top_heads_tuples.append(tuple(h))
        else:
            top_heads_tuples.append(h)
    
    # Convert changes dictionary keys to tuples if needed
    changes_1_tuples = {}
    for k, v in changes_1.items():
        if isinstance(k, np.ndarray):
            key = tuple(k.tolist())
        elif isinstance(k, list):
            key = tuple(k)
        else:
            key = k
        changes_1_tuples[key] = v
    
    changes_2_tuples = {}
    for k, v in changes_2.items():
        if isinstance(k, np.ndarray):
            key = tuple(k.tolist())
        elif isinstance(k, list):
            key = tuple(k)
        else:
            key = k
        changes_2_tuples[key] = v
    
    # Get common heads that exist in both models
    common_heads = [h for h in top_heads_tuples if (h in changes_1_tuples) and (h in changes_2_tuples)]
    
    if len(common_heads) == 0:
        print("Warning: No common heads found between the two models!")
        return
    
    print(f"Comparing {len(common_heads)} common heads...")
    
    # Individual head comparison (first 12 heads) - Before intervention only
    n_plot = min(12, len(common_heads))
    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    axes = axes.flatten()
    
    for idx, (layer, head) in enumerate(common_heads[:n_plot]):
        ax = axes[idx]
        acts_before_1 = changes_1_tuples[(layer, head)]['acts_before']
        acts_before_2 = changes_2_tuples[(layer, head)]['acts_before']
        
        # Use PCA for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        
        # Fit PCA on combined data from both models (before only)
        all_acts = np.vstack([acts_before_1, acts_before_2])
        pca.fit(all_acts)
        
        # Transform activations
        acts_before_1_2d = pca.transform(acts_before_1)
        acts_before_2_2d = pca.transform(acts_before_2)
        
        # Plot model 1 before
        ax.scatter(acts_before_1_2d[:, 0], acts_before_1_2d[:, 1], 
                  alpha=0.5, label=model_name_1, s=20, marker='o', color='blue')
        
        # Plot model 2 before
        ax.scatter(acts_before_2_2d[:, 0], acts_before_2_2d[:, 1], 
                  alpha=0.5, label=model_name_2, s=20, marker='s', color='red')
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'L{layer}H{head}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plot, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Representation Comparison (Before Intervention) - Top {n_plot} Heads', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}individual_comparison_alpha{alpha}_top{num_heads}.pdf'), 
                format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare activations between two models")
    parser.add_argument("--num_heads", type=int, default=18, help="Number of top heads")
    parser.add_argument("--alpha", type=float, default=5.0, help="Intervention strength (alpha)")
    parser.add_argument("--model_name_1", type=str, required=True, help="First model name")
    parser.add_argument("--model_name_2", type=str, required=True, help="Second model name")
    parser.add_argument("--dataset_name", type=str, default="toxigen_vicuna", help="Dataset name")
    parser.add_argument("--heads_path", type=str, required=True, help="Path to top heads file")
    parser.add_argument("--output_dir", type=str, default="./activation_plots_compare", help="Output directory for plots")
    parser.add_argument("--device_1", type=int, default=0, help="CUDA device for model 1")
    parser.add_argument("--device_2", type=int, default=0, help="CUDA device for model 2")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of prompts to use")
    parser.add_argument("--use_special_direction", action="store_true", help="Use special direction")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--fold", type=int, default=0, help="Fold number")
    
    args = parser.parse_args()
    
    device_1 = torch.device(f"cuda:{args.device_1}" if torch.cuda.is_available() else "cpu")
    device_2 = torch.device(f"cuda:{args.device_2}" if torch.cuda.is_available() else "cpu")
    
    # Load top heads
    print(f"Loading top heads from {args.heads_path}")
    top_heads = load_top_heads(args.heads_path)
    print(f"Loaded {len(top_heads)} heads: {top_heads[:5]}...")
    
    # Load model 1
    print(f"\nLoading model 1: {args.model_name_1}...")
    if args.model_name_1 in HF_NAMES:
        model_name_1 = HF_NAMES[args.model_name_1]
    else:
        model_name_1 = "/work/hdd/bcxt/yian3/toxic/models/" + args.model_name_1
    
    model_1 = AutoModelForCausalLM.from_pretrained(
        model_name_1,
        torch_dtype=torch.float16,
        device_map=f"cuda:{args.device_1}" if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True
    )
    model_1.eval()
    
    tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
    if tokenizer_1.pad_token is None:
        tokenizer_1.pad_token = tokenizer_1.eos_token
    
    # Load model 2
    print(f"\nLoading model 2: {args.model_name_2}...")
    if args.model_name_2 in HF_NAMES:
        model_name_2 = HF_NAMES[args.model_name_2]
    else:
        model_name_2 = "/work/hdd/bcxt/yian3/toxic/models/" + args.model_name_2
    
    model_2 = AutoModelForCausalLM.from_pretrained(
        model_name_2,
        torch_dtype=torch.float16,
        device_map=f"cuda:{args.device_2}" if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True
    )
    model_2.eval()
    
    tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
    if tokenizer_2.pad_token is None:
        tokenizer_2.pad_token = tokenizer_2.eos_token
    
    # Load interventions (using model 1's structure)
    num_heads_model_1 = model_1.config.num_attention_heads
    print(f"\nLoading interventions...")
    interventions_1 = load_interventions(
        top_heads, args.model_name_1, args.dataset_name, args.alpha,
        num_heads_model_1, args.use_special_direction, False, False,
        args.seed, args.fold
    )
    
    # For model 2, we'll use the same interventions structure
    # (assuming same head dimensions - may need adjustment for different models)
    num_heads_model_2 = model_2.config.num_attention_heads
    interventions_2 = load_interventions(
        top_heads, args.model_name_2, args.dataset_name, args.alpha,
        num_heads_model_2, args.use_special_direction, False, False,
        args.seed, args.fold
    )
    
    # Get test prompts
    print("\nLoading test prompts...")
    dataset_path = f'/work/hdd/bcxt/yian3/toxic/features/{args.dataset_name}_texts.json'
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            data = [json.loads(line) for line in f]
        prompts = [item.get('text', item.get('prompt', '')) for item in data[:args.num_samples]]
    else:
        prompts = [
            "What is the capital of France?",
            "Explain how photosynthesis works.",
            "What are the benefits of exercise?",
        ] * (args.num_samples // 3 + 1)
        prompts = prompts[:args.num_samples]
    
    # Collect activations for model 1
    print(f"\nCollecting activations for {args.model_name_1}...")
    activations_before_1, activations_after_1 = collect_activations_with_intervention(
        model_1, tokenizer_1, prompts, top_heads, interventions_1, args.alpha, device_1, args.num_samples
    )
    
    # Collect activations for model 2
    print(f"\nCollecting activations for {args.model_name_2}...")
    activations_before_2, activations_after_2 = collect_activations_with_intervention(
        model_2, tokenizer_2, prompts, top_heads, interventions_2, args.alpha, device_2, args.num_samples
    )
    
    # Compute changes
    print("\nComputing representation changes...")
    changes_1 = compute_representation_change(activations_before_1, activations_after_1)
    changes_2 = compute_representation_change(activations_before_2, activations_after_2)
    
    # Compare and plot
    print("\nGenerating comparison plots...")
    compare_models(changes_1, changes_2, top_heads, args.output_dir, args.alpha, args.num_heads,
                   args.model_name_1, args.model_name_2, args.dataset_name)
    
    print("Done!")


if __name__ == "__main__":
    main()

