#!/usr/bin/env python3
"""
Consolidate individual diff files into a single dictionary structure.
Converts files like L9_H5.npy into a dict format for easier loading.
"""

import os
import numpy as np
import pickle
import re
import argparse
from pathlib import Path

def consolidate_diffs(diffs_dir, output_path):
    """
    Consolidate individual diff files into a single dictionary.
    
    Args:
        diffs_dir: Path to directory containing L{layer}_H{head}.npy files
        output_path: Path to save the consolidated dictionary
    """
    diffs_dict = {}
    
    # Pattern to match L{layer}_H{head}.npy files
    pattern = re.compile(r'L(\d+)_H(\d+)\.npy')
    
    print(f"Scanning directory: {diffs_dir}")
    
    for filename in os.listdir(diffs_dir):
        match = pattern.match(filename)
        if match:
            layer = int(match.group(1))
            head = int(match.group(2))
            
            file_path = os.path.join(diffs_dir, filename)
            print(f"Loading {filename} -> Layer {layer}, Head {head}")
            
            # Load the diff data
            diff_data = np.load(file_path)
            
            # Store in dictionary with (layer, head) as key
            diffs_dict[(layer, head)] = diff_data
            
            print(f"  Shape: {diff_data.shape}, dtype: {diff_data.dtype}")
    
    # Save the consolidated dictionary
    print(f"\nSaving consolidated diffs to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(diffs_dict, f)
    
    print(f"Saved {len(diffs_dict)} layer-head combinations")
    print(f"Keys: {sorted(diffs_dict.keys())}")
    
    return diffs_dict

def load_consolidated_diffs(diffs_path):
    """Load the consolidated diffs dictionary."""
    with open(diffs_path, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="Consolidate individual diff files into a single dictionary")
    parser.add_argument("--dataset_name", type=str, default="toxigen_vicuna", 
                       help="Dataset name")
    parser.add_argument("--model_name", type=str, default="llama3_8B", 
                       help="Model name")
    parser.add_argument("--output_dir", type=str, default="./local_store",
                       help="Output directory for local store")
    parser.add_argument("--heads_path", type=str, default=None,
                       help="Path to pre-selected heads numpy file (used to determine pns suffix)")
    
    args = parser.parse_args()
    
    # Determine if using PNS heads from the heads_path
    use_pns = "True" if args.heads_path and "True" in args.heads_path else False
    pns_suffix = "pns" if use_pns else "no_pns"
    
    # Create specific output directory structure (same as build_local_store.py)
    specific_output_dir = os.path.join(
        args.output_dir, 
        f"{args.model_name}_{args.dataset_name}_{pns_suffix}"
    )
    
    # Paths
    base_dir = specific_output_dir
    diffs_dir = os.path.join(base_dir, "diffs")
    output_path = os.path.join(base_dir, "diffs_consolidated.pkl")
    
    print(f"Using base directory: {base_dir}")
    print(f"Diffs directory: {diffs_dir}")
    print(f"Output path: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Consolidate the diffs
    diffs_dict = consolidate_diffs(diffs_dir, output_path)
    
    # Verify the consolidation
    print("\nVerification:")
    loaded_diffs = load_consolidated_diffs(output_path)
    print(f"Loaded {len(loaded_diffs)} entries")
    
    # Show some examples
    for i, (layer, head) in enumerate(sorted(loaded_diffs.keys())):
        if i < 5:  # Show first 5
            print(f"  ({layer}, {head}): shape {loaded_diffs[(layer, head)].shape}")
        elif i == 5:
            print(f"  ... and {len(loaded_diffs) - 5} more")

if __name__ == "__main__":
    main()

"""
python consolidate_diffs.py \
    --dataset_name toxigen_vicuna \
    --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 \
    --output_dir /work/hdd/bcxt/yian3/toxic/local_store_toxigen \
    --heads_path /work/hdd/bcxt/yian3/toxic/features/heads/True_llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5_toxigen_vicuna_seed_2_top_72_heads_fold_0.npy
"""