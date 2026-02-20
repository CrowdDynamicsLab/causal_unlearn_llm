#!/usr/bin/env python3
"""
Analyze dataset statistics: average text lengths and toxic scores.

This script calculates:
- Average length of text, non_toxic_text, and toxic_text
- Average toxic score for these columns

For datasets: paradetox, toxigen_vicuna, hate_vicuna

python analyze_dataset_stats.py --datasets paradetox toxigen_vicuna hate_vicuna
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Try to import Detoxify for computing toxic scores
try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    DETOXIFY_AVAILABLE = False
    print("Warning: Detoxify not available. Toxic scores will only be extracted from dataset if present.")


def calculate_text_length(text):
    """Calculate length of text (word count and character count)."""
    if not text or not isinstance(text, str):
        return 0, 0
    words = text.split()
    return len(words), len(text)


def compute_toxic_score(text, detoxify_model=None):
    """Compute toxic score for a text using Detoxify if available."""
    if detoxify_model is not None:
        try:
            results = detoxify_model.predict(text)
            # Return the 'toxicity' score (or 'toxic' if that's the key)
            return results.get('toxicity', results.get('toxic', 0.0))
        except Exception as e:
            print(f"Warning: Error computing toxic score: {e}")
            return None
    return None


def analyze_dataset(dataset_name, dataset_path, compute_scores=False):
    """Analyze a single dataset."""
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset file not found: {dataset_path}")
        return None
    
    print(f"\nAnalyzing {dataset_name}...")
    print(f"Loading from: {dataset_path}")
    
    # Load data - match the exact pattern from validate_2fold_toxic.py
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data)} examples")
    
    # Show sample keys to understand structure
    if len(data) > 0:
        print(f"Sample keys: {list(data[0].keys())}")
    
    # Initialize Detoxify model if needed
    detoxify_model = None
    if compute_scores and DETOXIFY_AVAILABLE:
        print("Loading Detoxify model for toxic score computation...")
        try:
            detoxify_model = Detoxify('original')
            print("Detoxify model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load Detoxify model: {e}")
            compute_scores = False
    
    # Initialize statistics
    stats = {
        'text': {'word_lengths': [], 'char_lengths': [], 'toxic_scores': []},
        'non_toxic_text': {'word_lengths': [], 'char_lengths': [], 'toxic_scores': []},
        'toxic_text': {'word_lengths': [], 'char_lengths': [], 'toxic_scores': []},
    }
    
    # Process each example
    for item in tqdm(data, desc=f"Processing {dataset_name}"):
        # Process 'text' field
        if 'text' in item:
            text = item['text']
            if text and isinstance(text, str):
                word_len, char_len = calculate_text_length(text)
                stats['text']['word_lengths'].append(word_len)
                stats['text']['char_lengths'].append(char_len)
                # Try to get toxic score for text
                toxic_score = item.get('toxicity_score', item.get('toxicity', None))
                if toxic_score is not None:
                    try:
                        stats['text']['toxic_scores'].append(float(toxic_score))
                    except (ValueError, TypeError):
                        # If compute_scores is enabled, we'll compute it later
                        if not compute_scores:
                            pass  # Skip if not computing
                else:
                    # No score in dataset - will compute if compute_scores is True
                    pass
        
        # Process 'non_toxic_text' field
        if 'non_toxic_text' in item:
            text = item['non_toxic_text']
            if text and isinstance(text, str) and text.strip():
                word_len, char_len = calculate_text_length(text)
                stats['non_toxic_text']['word_lengths'].append(word_len)
                stats['non_toxic_text']['char_lengths'].append(char_len)
                # Try to get toxic score
                toxic_score = item.get('non_toxic_toxicity_score', item.get('non_toxic_toxicity', None))
                if toxic_score is not None:
                    try:
                        stats['non_toxic_text']['toxic_scores'].append(float(toxic_score))
                    except (ValueError, TypeError):
                        if not compute_scores:
                            pass
        
        # Process 'toxic_text' field
        if 'toxic_text' in item:
            text = item['toxic_text']
            if text and isinstance(text, str) and text.strip():
                word_len, char_len = calculate_text_length(text)
                stats['toxic_text']['word_lengths'].append(word_len)
                stats['toxic_text']['char_lengths'].append(char_len)
                # Try to get toxic score
                toxic_score = item.get('toxic_toxicity_score', item.get('toxic_toxicity', None))
                if toxic_score is not None:
                    try:
                        stats['toxic_text']['toxic_scores'].append(float(toxic_score))
                    except (ValueError, TypeError):
                        if not compute_scores:
                            pass
    
    # Compute toxic scores for texts that don't have them
    if compute_scores and detoxify_model:
        print(f"Computing toxic scores for texts missing scores...")
        # Re-process data to compute scores for texts without scores
        for item in tqdm(data, desc="Computing missing toxic scores"):
            # Process 'text' field
            if 'text' in item:
                text = item['text']
                if text and isinstance(text, str):
                    # Check if we already have a score for this text
                    toxic_score = item.get('toxicity_score', item.get('toxicity', None))
                    if toxic_score is None:
                        score = compute_toxic_score(text, detoxify_model)
                        if score is not None:
                            stats['text']['toxic_scores'].append(score)
            
            # Process 'non_toxic_text' field
            if 'non_toxic_text' in item:
                text = item['non_toxic_text']
                if text and isinstance(text, str) and text.strip():
                    toxic_score = item.get('non_toxic_toxicity_score', item.get('non_toxic_toxicity', None))
                    if toxic_score is None:
                        score = compute_toxic_score(text, detoxify_model)
                        if score is not None:
                            stats['non_toxic_text']['toxic_scores'].append(score)
            
            # Process 'toxic_text' field
            if 'toxic_text' in item:
                text = item['toxic_text']
                if text and isinstance(text, str) and text.strip():
                    toxic_score = item.get('toxic_toxicity_score', item.get('toxic_toxicity', None))
                    if toxic_score is None:
                        score = compute_toxic_score(text, detoxify_model)
                        if score is not None:
                            stats['toxic_text']['toxic_scores'].append(score)
    
    # Calculate averages
    results = {}
    for field in ['text', 'non_toxic_text', 'toxic_text']:
        results[field] = {
            'avg_word_length': np.mean(stats[field]['word_lengths']) if stats[field]['word_lengths'] else 0,
            'avg_char_length': np.mean(stats[field]['char_lengths']) if stats[field]['char_lengths'] else 0,
            'count': len(stats[field]['word_lengths']),
            'avg_toxic_score': np.mean(stats[field]['toxic_scores']) if stats[field]['toxic_scores'] else None,
            'toxic_score_count': len(stats[field]['toxic_scores'])
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset statistics")
    parser.add_argument("--datasets", nargs='+', 
                       default=['paradetox', 'toxigen_vicuna', 'hate_vicuna'],
                       help="List of datasets to analyze")
    parser.add_argument("--base_path", type=str,
                       default="/work/hdd/bcxt/yian3/toxic/features",
                       help="Base path for dataset files")
    parser.add_argument("--output_file", type=str,
                       default="dataset_stats.csv",
                       help="Output file path to save results (CSV or JSON). If not specified, results are only printed.")
    parser.add_argument("--no_compute_toxic_scores", dest="compute_toxic_scores", action="store_false",
                       help="Don't compute toxic scores, only use scores from dataset")
    parser.set_defaults(compute_toxic_scores=True)
    
    args = parser.parse_args()
    
    # Dataset file paths
    dataset_paths = {
        'paradetox': os.path.join(args.base_path, 'paradetox_texts.json'),
        'toxigen_vicuna': os.path.join(args.base_path, 'toxigen_vicuna_texts.json'),
        'hate_vicuna': os.path.join(args.base_path, 'hate_vicuna_texts.json'),
    }
    
    all_results = {}
    
    for dataset_name in args.datasets:
        if dataset_name not in dataset_paths:
            print(f"Warning: Unknown dataset {dataset_name}, skipping...")
            continue
        
        results = analyze_dataset(dataset_name, dataset_paths[dataset_name], 
                                  compute_scores=args.compute_toxic_scores)
        if results:
            all_results[dataset_name] = results
    
    # Print results
    print("\n" + "="*80)
    print("DATASET STATISTICS SUMMARY")
    print("="*80)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print("-" * 80)
        
        for field in ['text', 'non_toxic_text', 'toxic_text']:
            field_results = results[field]
            print(f"\n  {field}:")
            print(f"    Count: {field_results['count']}")
            print(f"    Average word length: {field_results['avg_word_length']:.2f}")
            print(f"    Average character length: {field_results['avg_char_length']:.2f}")
            if field_results['avg_toxic_score'] is not None:
                print(f"    Average toxic score: {field_results['avg_toxic_score']:.4f} (n={field_results['toxic_score_count']})")
            else:
                print(f"    Average toxic score: N/A (no scores found)")
    
    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Dataset':<20} {'Field':<15} {'Avg Words':<12} {'Avg Chars':<12} {'Avg Tox Score':<15} {'Count':<10}")
    print("-" * 80)
    
    for dataset_name, results in all_results.items():
        for field in ['text', 'non_toxic_text', 'toxic_text']:
            field_results = results[field]
            tox_score_str = f"{field_results['avg_toxic_score']:.4f}" if field_results['avg_toxic_score'] is not None else "N/A"
            print(f"{dataset_name:<20} {field:<15} {field_results['avg_word_length']:<12.2f} "
                  f"{field_results['avg_char_length']:<12.2f} {tox_score_str:<15} {field_results['count']:<10}")
    
    print("="*80)
    
    # Save results if output file is specified
    if args.output_file:
        # Prepare data for saving
        rows = []
        for dataset_name, results in all_results.items():
            for field in ['text', 'non_toxic_text', 'toxic_text']:
                field_results = results[field]
                rows.append({
                    'dataset': dataset_name,
                    'field': field,
                    'avg_word_length': field_results['avg_word_length'],
                    'avg_char_length': field_results['avg_char_length'],
                    'avg_toxic_score': field_results['avg_toxic_score'] if field_results['avg_toxic_score'] is not None else None,
                    'count': field_results['count'],
                    'toxic_score_count': field_results['toxic_score_count']
                })
        
        df = pd.DataFrame(rows)
        
        # Save based on file extension
        if args.output_file.endswith('.csv'):
            df.to_csv(args.output_file, index=False)
            print(f"\n✅ Results saved to CSV: {args.output_file}")
        elif args.output_file.endswith('.json'):
            df.to_json(args.output_file, orient='records', indent=2)
            print(f"\n✅ Results saved to JSON: {args.output_file}")
        else:
            # Default to CSV
            csv_path = args.output_file + '.csv'
            df.to_csv(csv_path, index=False)
            print(f"\n✅ Results saved to CSV: {csv_path}")
        
        # Also save raw results as JSON for programmatic access
        json_path = args.output_file.replace('.csv', '.json') if args.output_file.endswith('.csv') else args.output_file + '.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✅ Raw results saved to JSON: {json_path}")


if __name__ == "__main__":
    main()

