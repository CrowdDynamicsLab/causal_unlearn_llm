#!/usr/bin/env python3
"""
Plot activations before and after intervention for top heads.

This script:
1. Loads top heads (e.g., top 18 heads)
2. Loads model and interventions with specified alpha
3. Collects activations before and after intervention
4. Plots representation changes for each head

Usage example:
    python plot_activations.py --num_heads 18 --alpha 5.0 \
        --model_name llama3_8B --dataset_name toxigen_vicuna \
        --heads_path /work/hdd/bcxt/yian3/toxic/features/heads/True_llama3_8B_toxigen_vicuna_seed_0_top_18_heads_fold_0.npy \
        --output_dir ./activation_plots --num_samples 50
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
from utils_toxic import (
    layer_head_to_flattened_idx,
    flattened_idx_to_layer_head,
    get_interventions_dict,
    get_com_directions,
    get_special_directions,
    get_matrix_directions,
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


def load_top_heads(heads_path):
    """Load top heads from numpy file."""
    if not os.path.exists(heads_path):
        raise FileNotFoundError(f"Heads file not found: {heads_path}")
    heads = np.load(heads_path, allow_pickle=True)
    if isinstance(heads, np.ndarray) and heads.dtype == object:
        heads = heads.item()
    if isinstance(heads, dict):
        # If it's a dict, try to extract the heads list
        heads = heads.get('heads', heads.get('top_heads', list(heads.values())[0] if heads else []))
    if isinstance(heads, list):
        # Convert to list of tuples if needed
        if len(heads) > 0 and isinstance(heads[0], (list, np.ndarray)):
            heads = [tuple(h) if isinstance(h, (list, np.ndarray)) else h for h in heads]
    return heads


def load_interventions(top_heads, model_name, dataset_name, alpha, num_heads, 
                       use_special_direction=False, use_center_of_mass=False, 
                       use_mat_direction=False, seed=0, fold=0):
    """Load or create interventions for the top heads."""
    # Try to load precomputed interventions
    interventions_path = f"/work/hdd/bcxt/yian3/toxic/features/interventions/{model_name}_{dataset_name}_top{len(top_heads)}_alpha{alpha}_fold{fold}.pkl"
    
    if os.path.exists(interventions_path):
        print(f"Loading interventions from {interventions_path}")
        with open(interventions_path, 'rb') as f:
            import pickle
            return pickle.load(f)
    
    # Otherwise, we need to load activations and compute interventions
    print("Interventions file not found. You may need to run validate_2fold_toxic.py first.")
    print("For now, creating dummy interventions structure...")
    
    # Create a basic intervention structure
    interventions = {}
    for layer, head in top_heads:
        layer_name = f"model.layers.{layer}.self_attn.o_proj"
        if layer_name not in interventions:
            interventions[layer_name] = []
        # Create a dummy direction (will be replaced if you have actual data)
        direction = np.random.normal(size=(128,))  # Default head dim
        direction = direction / np.linalg.norm(direction)
        interventions[layer_name].append((head, direction, 1.0))
    
    # Sort by head index
    for layer_name in interventions:
        interventions[layer_name] = sorted(interventions[layer_name], key=lambda x: x[0])
    
    return interventions


class ActivationCollector:
    """Collector that captures both before and after activations."""
    collect_state = True
    collect_action = False
    
    def __init__(self, head_idx, num_heads_model):
        self.head_idx = head_idx
        self.num_heads_model = num_heads_model
        self.activation_before = None
        self.activation_after = None
    
    def reset(self):
        self.activation_before = None
        self.activation_after = None
    
    def __call__(self, b, s):
        # Capture before intervention
        final_token = b[0, -1].detach().clone()
        head_dim = final_token.shape[-1] // self.num_heads_model
        if self.head_idx < self.num_heads_model:
            self.activation_before = final_token.reshape(self.num_heads_model, head_dim)[self.head_idx].cpu().numpy()
        else:
            self.activation_before = final_token.cpu().numpy()
        return b


class InterventionCollector:
    """Collector that captures activations before and after intervention."""
    collect_state = True
    collect_action = True
    
    def __init__(self, head_idx, num_heads_model, direction, multiplier):
        self.head_idx = head_idx
        self.num_heads_model = num_heads_model
        # Store direction as numpy, will convert to tensor on device when needed
        if isinstance(direction, torch.Tensor):
            self.direction = direction.cpu().numpy()
        else:
            self.direction = np.array(direction)
        self.multiplier = multiplier
        self.activation_before = None
        self.activation_after = None
    
    def reset(self):
        self.activation_before = None
        self.activation_after = None
    
    def __call__(self, b, s):
        # Capture before intervention
        final_token = b[0, -1].detach().clone()
        head_dim = final_token.shape[-1] // self.num_heads_model
        if self.head_idx < self.num_heads_model:
            self.activation_before = final_token.reshape(self.num_heads_model, head_dim)[self.head_idx].detach().clone().cpu().numpy()
        else:
            self.activation_before = final_token.detach().clone().cpu().numpy()
        
        # Apply intervention
        # Convert direction to tensor on the correct device
        action = torch.tensor(self.direction, dtype=b.dtype, device=b.device)
        head_dim_full = final_token.shape[-1]
        
        # Handle different direction formats:
        # 1. If direction is (head_dim, head_dim) - it's a matrix, apply to head-specific slice
        # 2. If direction is (head_dim,) - it's a vector for this head
        # 3. If direction is (head_dim_full,) - it's a full direction vector
        
        if len(action.shape) == 2:
            # Matrix direction: (head_dim, head_dim)
            head_start = self.head_idx * head_dim
            head_end = head_start + head_dim
            head_activation = b[0, -1, head_start:head_end]  # [head_dim]
            # Apply matrix transformation: head_activation @ direction.T
            transformed = head_activation @ action.T  # [head_dim]
            b[0, -1, head_start:head_end] = b[0, -1, head_start:head_end] + transformed * self.multiplier
        elif len(action.shape) == 1:
            if action.shape[0] == head_dim:
                # Head-specific vector direction
                head_start = self.head_idx * head_dim
                head_end = head_start + head_dim
                b[0, -1, head_start:head_end] = b[0, -1, head_start:head_end] + action * self.multiplier
            elif action.shape[0] == head_dim_full:
                # Full direction vector
                b[0, -1] = b[0, -1] + action * self.multiplier
            else:
                # Unknown shape, try to apply anyway
                b[0, -1] = b[0, -1] + action * self.multiplier
        
        # Capture after intervention
        final_token_after = b[0, -1].detach().clone()
        if self.head_idx < self.num_heads_model:
            self.activation_after = final_token_after.reshape(self.num_heads_model, head_dim)[self.head_idx].detach().clone().cpu().numpy()
        else:
            self.activation_after = final_token_after.detach().clone().cpu().numpy()
        
        return b


def collect_activations_with_intervention(model, tokenizer, prompts, top_heads, 
                                          interventions, alpha, device, num_samples=50):
    """
    Collect activations before and after intervention for each head.
    
    Returns:
        activations_before: dict of {(layer, head): [activations]}
        activations_after: dict of {(layer, head): [activations]}
    """
    activations_before = {(l, h): [] for l, h in top_heads}
    activations_after = {(l, h): [] for l, h in top_heads}
    
    # Limit number of prompts for faster processing
    prompts = prompts[:num_samples] if len(prompts) > num_samples else prompts
    
    num_heads_model = model.config.num_attention_heads
    
    print(f"Collecting activations for {len(prompts)} prompts...")
    
    for prompt in tqdm(prompts, desc="Processing prompts"):
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        
        # Set up collectors for each head
        collectors = {}
        pv_config = []
        
        for layer, head in top_heads:
            layer_name = f"model.layers.{layer}.self_attn.o_proj"
            if layer_name in interventions:
                # Find the direction for this head
                direction = None
                for h, d, std in interventions[layer_name]:
                    if h == head:
                        direction = d
                        break
                
                if direction is not None:
                    # Convert direction to numpy if needed, then to tensor
                    if isinstance(direction, torch.Tensor):
                        direction_np = direction.cpu().numpy()
                    else:
                        direction_np = np.array(direction)
                    collector = InterventionCollector(head, num_heads_model, direction_np, alpha)
                    collectors[(layer, head)] = collector
                    pv_config.append({
                        "component": f"model.layers[{layer}].self_attn.o_proj.input",
                        "intervention": wrapper(collector),
                    })
                else:
                    # No intervention, just collect before
                    collector = ActivationCollector(head, num_heads_model)
                    collectors[(layer, head)] = collector
                    pv_config.append({
                        "component": f"model.layers[{layer}].self_attn.o_proj.input",
                        "intervention": wrapper(collector),
                    })
            else:
                # No intervention, just collect before
                collector = ActivationCollector(head, num_heads_model)
                collectors[(layer, head)] = collector
                pv_config.append({
                    "component": f"model.layers[{layer}].self_attn.o_proj.input",
                    "intervention": wrapper(collector),
                })
        
        # Run model
        if pv_config:
            collected_model = pv.IntervenableModel(pv_config, model)
            with torch.no_grad():
                _ = collected_model({"input_ids": input_ids})
            
            # Extract activations
            for (layer, head), collector in collectors.items():
                if isinstance(collector, InterventionCollector):
                    if collector.activation_before is not None:
                        activations_before[(layer, head)].append(collector.activation_before)
                    if collector.activation_after is not None:
                        activations_after[(layer, head)].append(collector.activation_after)
                elif isinstance(collector, ActivationCollector):
                    if collector.activation_before is not None:
                        activations_before[(layer, head)].append(collector.activation_before)
                        # For heads without intervention, after = before
                        activations_after[(layer, head)].append(collector.activation_before)
                collector.reset()
    
    return activations_before, activations_after


def compute_representation_change(activations_before, activations_after):
    """Compute representation change metrics."""
    changes = {}
    for (layer, head) in activations_before.keys():
        acts_before = np.array(activations_before[(layer, head)])
        acts_after = np.array(activations_after[(layer, head)])
        
        if len(acts_before) == 0 or len(acts_after) == 0:
            continue
        
        # Compute mean change
        mean_change = np.mean(acts_after - acts_before, axis=0)
        
        # Compute L2 norm of change
        l2_change = np.linalg.norm(acts_after - acts_before, axis=1)
        mean_l2_change = np.mean(l2_change)
        
        # Compute cosine similarity
        cosine_sims = []
        for i in range(len(acts_before)):
            if np.linalg.norm(acts_before[i]) > 0 and np.linalg.norm(acts_after[i]) > 0:
                cos_sim = np.dot(acts_before[i], acts_after[i]) / (
                    np.linalg.norm(acts_before[i]) * np.linalg.norm(acts_after[i])
                )
                cosine_sims.append(cos_sim)
        
        changes[(layer, head)] = {
            'mean_change': mean_change,
            'mean_l2_change': mean_l2_change,
            'mean_cosine_sim': np.mean(cosine_sims) if cosine_sims else 0.0,
            'acts_before': acts_before,
            'acts_after': acts_after,
        }
    
    return changes


def plot_activation_changes(changes, top_heads, output_dir, alpha, num_heads, model_name=None, dataset_name=None):
    """Plot activation changes for all heads."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename prefix with model name and dataset name if provided
    filename_parts = []
    if model_name:
        # Clean model name for filename (remove special characters)
        model_name_clean = model_name.replace('/', '_').replace('-', '_')
        filename_parts.append(model_name_clean)
    if dataset_name:
        # Clean dataset name for filename
        dataset_name_clean = dataset_name.replace('/', '_').replace('-', '_')
        filename_parts.append(dataset_name_clean)
    
    if filename_parts:
        filename_prefix = "_".join(filename_parts) + "_"
    else:
        filename_prefix = ""
    
    # Create summary plots
    n_heads = len(top_heads)
    n_cols = 4
    n_rows = (n_heads + n_cols - 1) // n_cols
    
    # Plot 1: Mean L2 change per head
    # fig, ax = plt.subplots(figsize=(12, 6))
    # head_labels = [f"L{l}H{h}" for l, h in top_heads]
    # l2_changes = [changes[(l, h)]['mean_l2_change'] for l, h in top_heads if (l, h) in changes]
    # head_labels_filtered = [f"L{l}H{h}" for l, h in top_heads if (l, h) in changes]
    
    # ax.bar(range(len(l2_changes)), l2_changes)
    # ax.set_xticks(range(len(head_labels_filtered)))
    # ax.set_xticklabels(head_labels_filtered, rotation=45, ha='right')
    # ax.set_xlabel('Head (Layer, Head)')
    # ax.set_ylabel('Mean L2 Change')
    # ax.set_title(f'Representation Change (L2 Norm) - Top {num_heads} Heads, Alpha={alpha}')
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, f'{filename_prefix}activation_changes_l2_alpha{alpha}_top{num_heads}.pdf'), format='pdf', bbox_inches='tight')
    # plt.close()
    
    # Plot 2: Cosine similarity per head
    # fig, ax = plt.subplots(figsize=(12, 6))
    # cosine_sims = [changes[(l, h)]['mean_cosine_sim'] for l, h in top_heads if (l, h) in changes]
    
    # ax.bar(range(len(cosine_sims)), cosine_sims)
    # ax.set_xticks(range(len(head_labels_filtered)))
    # ax.set_xticklabels(head_labels_filtered, rotation=45, ha='right')
    # ax.set_xlabel('Head (Layer, Head)')
    # ax.set_ylabel('Mean Cosine Similarity')
    # ax.set_title(f'Representation Similarity - Top {num_heads} Heads, Alpha={alpha}')
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, f'{filename_prefix}activation_similarity_alpha{alpha}_top{num_heads}.pdf'), format='pdf', bbox_inches='tight')
    # plt.close()
    
    # Plot 3: Individual head plots (first 12 heads)
    n_plot = min(12, len(top_heads))
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (layer, head) in enumerate(top_heads[:n_plot]):
        if (layer, head) not in changes:
            continue
        
        ax = axes[idx]
        acts_before = changes[(layer, head)]['acts_before']
        acts_after = changes[(layer, head)]['acts_after']
        
        # Plot mean activation before and after
        mean_before = np.mean(acts_before, axis=0)
        mean_after = np.mean(acts_after, axis=0)
        
        # Use PCA or first few dimensions for visualization
        if len(mean_before) > 2:
            # Take first 2 dimensions or use PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            acts_before_2d = pca.fit_transform(acts_before)
            acts_after_2d = pca.transform(acts_after)
            ax.scatter(acts_before_2d[:, 0], acts_before_2d[:, 1], 
                      alpha=0.5, label='Before', s=20)
            ax.scatter(acts_after_2d[:, 0], acts_after_2d[:, 1], 
                      alpha=0.5, label='After', s=20)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
        else:
            ax.scatter(mean_before[0], mean_before[1] if len(mean_before) > 1 else 0,
                      marker='o', s=100, label='Before')
            ax.scatter(mean_after[0], mean_after[1] if len(mean_after) > 1 else 0,
                      marker='x', s=100, label='After')
            ax.set_xlabel('Dim 0')
            ax.set_ylabel('Dim 1')
        
        ax.set_title(f'L{layer}H{head}\nL2: {changes[(layer, head)]["mean_l2_change"]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plot, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Activation Changes - Top {n_plot} Heads, Alpha={alpha}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'activation_changes_{filename_prefix}_alpha{alpha}_top{num_heads}.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot activations before and after intervention")
    parser.add_argument("--num_heads", type=int, default=18, help="Number of top heads")
    parser.add_argument("--alpha", type=float, default=5.0, help="Intervention strength (alpha)")
    parser.add_argument("--model_name", type=str, default="llama3_8B", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="toxigen_vicuna", help="Dataset name")
    parser.add_argument("--heads_path", type=str, 
                       default="/work/hdd/bcxt/yian3/toxic/features/heads/True_llama3_8B_toxigen_vicuna_seed_0_top_18_heads_fold_0.npy",
                       help="Path to top heads file")
    parser.add_argument("--output_dir", type=str, default="./activation_plots", help="Output directory for plots")
    parser.add_argument("--device", type=int, default=0, help="CUDA device")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of prompts to use")
    parser.add_argument("--use_special_direction", action="store_true", help="Use special direction")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--fold", type=int, default=0, help="Fold number")
    
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # Load top heads
    print(f"Loading top heads from {args.heads_path}")
    top_heads = load_top_heads(args.heads_path)
    print(f"Loaded {len(top_heads)} heads: {top_heads[:5]}...")
    
    # Load model
    print(f"Loading model {args.model_name}...")
    if args.model_name in HF_NAMES:
        model_name = HF_NAMES[args.model_name]
    else:
        model_name = "/work/hdd/bcxt/yian3/toxic/models/" + args.model_name
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=f"cuda:{args.device}",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load interventions
    num_heads_model = model.config.num_attention_heads
    print(f"Loading interventions...")
    interventions = load_interventions(
        top_heads, args.model_name, args.dataset_name, args.alpha,
        num_heads_model, args.use_special_direction, False, False,
        args.seed, args.fold
    )
    
    # Get some test prompts
    print("Loading test prompts...")
    dataset_path = f'/work/hdd/bcxt/yian3/toxic/features/{args.dataset_name}_texts.json'
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            data = [json.loads(line) for line in f]
        prompts = [item.get('text', item.get('prompt', '')) for item in data[:args.num_samples]]
    else:
        # Fallback prompts
        prompts = [
            "What is the capital of France?",
            "Explain how photosynthesis works.",
            "What are the benefits of exercise?",
        ] * (args.num_samples // 3 + 1)
        prompts = prompts[:args.num_samples]
    
    # Collect activations
    print("Collecting activations...")
    activations_before, activations_after = collect_activations_with_intervention(
        model, tokenizer, prompts, top_heads, interventions, args.alpha, device, args.num_samples
    )
    
    # Compute changes
    print("Computing representation changes...")
    changes = compute_representation_change(activations_before, activations_after)
    
    # Plot
    print("Generating plots...")
    plot_activation_changes(changes, top_heads, args.output_dir, args.alpha, args.num_heads, args.model_name, args.dataset_name)
    
    print("Done!")


if __name__ == "__main__":
    main()

"""
python plot_activations.py --num_heads 36 --alpha 10.0 \
    --model_name vicuna_7B --dataset_name paradetox \
    --heads_path /work/hdd/bcxt/yian3/toxic/features/heads/True_vicuna_7B_paradetox_seed_2_top_72_heads_fold_0.npy \
    --output_dir ./activation_plots --num_samples 50

python plot_activations.py --num_heads 36 --alpha 5.0 \
    --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 --dataset_name toxigen_vicuna \
    --heads_path /work/hdd/bcxt/yian3/toxic/features/heads/False_llama3_8B_toxigen_vicuna_seed_2_top_72_heads_fold_0.npy \
    --output_dir ./activation_plots --num_samples 50
"""