#!/usr/bin/env python3
"""
Script to create train/test splits for datasets.
Usage: python create_splits.py --dataset_name hate_vicuna --num_fold 2 --seed 2
"""

import numpy as np
import pandas as pd
import json
import argparse
import os

def create_splits(dataset_name, num_fold=2, seed=2):
    """Create train/test splits for a given dataset."""
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Load dataset based on name
    if dataset_name == "hate_vicuna":
        hate_path = f'/work/hdd/bcxt/yian3/toxic/features/{dataset_name}_texts.json'
        with open(hate_path) as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
    elif dataset_name == "toxigen_vicuna":
        toxigen_path = f'/work/hdd/bcxt/yian3/toxic/features/{dataset_name}_texts.json'
        with open(toxigen_path) as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
    elif dataset_name == "paradetox":
        paradetox_path = f'/work/hdd/bcxt/yian3/toxic/features/{dataset_name}_texts.json'
        with open(paradetox_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Loaded {len(df)} examples from {dataset_name}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create splits directory if it doesn't exist
    os.makedirs('splits', exist_ok=True)
    
    # Create folds using numpy array_split (same as in validate_2fold_toxic.py)
    fold_idxs = np.array_split(np.arange(len(df)), num_fold)
    
    # Create train/test splits for each fold
    for i in range(num_fold):
        # Test set is fold i
        test_idxs = fold_idxs[i]
        
        # Train set is all other folds
        train_idxs = np.concatenate([fold_idxs[j] for j in range(num_fold) if j != i])
        
        # Save train and test splits
        train_df = df.iloc[train_idxs].reset_index(drop=True)
        test_df = df.iloc[test_idxs].reset_index(drop=True)
        
        train_path = f'splits/{dataset_name}_fold_{i}_train_seed_{seed}.csv'
        test_path = f'splits/{dataset_name}_fold_{i}_test_seed_{seed}.csv'
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Fold {i}:")
        print(f"  Train: {len(train_df)} examples -> {train_path}")
        print(f"  Test:  {len(test_df)} examples -> {test_path}")
    
    print(f"\n✅ Successfully created {num_fold} folds for {dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create train/test splits for datasets')
    parser.add_argument('--dataset_name', type=str, required=True, 
                       help='Dataset name (e.g., hate_vicuna, toxigen_vicuna, paradetox)')
    parser.add_argument('--num_fold', type=int, default=2, 
                       help='Number of folds (default: 2)')
    parser.add_argument('--seed', type=int, default=2, 
                       help='Random seed (default: 2)')
    
    args = parser.parse_args()
    create_splits(args.dataset_name, args.num_fold, args.seed)

