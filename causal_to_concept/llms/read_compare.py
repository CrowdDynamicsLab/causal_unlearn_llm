#!/usr/bin/env python3
"""
Simple CSV File Comparison Tool

This script compares two CSV files and prints:
- Mean toxicity (toxicity_gen column) for both files
- Mean toxicity (toxicity_text column) for both files
- Mean perplexity for both files
- Mean scores for any "{model} toxic" columns found in the files
- Mean scores for any "{model} sense" columns found in the files
"""

import argparse 
from pathlib import Path
import pandas as pd
import numpy as np


def find_perplexity_column(columns, specified_col=None):
    """Find perplexity column in the dataframe."""
    if specified_col:
        if specified_col in columns:
            return specified_col
        return None
    
    # Auto-detect perplexity columns
    ppl_candidates = [col for col in columns if 'ppl' in col.lower() or 'perplexity' in col.lower()]
    
    if len(ppl_candidates) >= 1:
        return ppl_candidates[0]
    
    return None


def calculate_mean_toxicity(df, col_name):
    """Calculate mean toxicity from specified column."""
    if col_name not in df.columns:
        return None
    
    tox_data = pd.to_numeric(df[col_name], errors="coerce").dropna()
    if tox_data.empty:
        return None
    
    return float(tox_data.mean())


def calculate_mean_perplexity(df, ppl_col=None, max_ppl=None):
    """Calculate mean perplexity, optionally filtering out values above max_ppl threshold."""
    auto_ppl_col = find_perplexity_column(df.columns.tolist(), ppl_col)
    if not auto_ppl_col:
        return None
    
    ppl_data = pd.to_numeric(df[auto_ppl_col], errors="coerce").dropna()
    if ppl_data.empty:
        return None
    
    # Filter out perplexity scores above threshold if max_ppl is specified
    if max_ppl is not None:
        ppl_filtered = ppl_data[ppl_data <= max_ppl]
        if ppl_filtered.empty:
            return None
        return float(ppl_filtered.mean())
    else:
        # Use regular average without filtering
        return float(ppl_data.mean())


def find_model_columns(df, suffix):
    """Find columns that end with the given suffix (e.g., ' toxic' or ' sense')."""
    matching_cols = [col for col in df.columns if col.endswith(suffix)]
    return matching_cols


def calculate_mean_for_columns(df, column_names):
    """Calculate mean for multiple columns, returning a dict of {column_name: mean_value}."""
    results = {}
    for col_name in column_names:
        if col_name in df.columns:
            data = pd.to_numeric(df[col_name], errors="coerce").dropna()
            if not data.empty:
                results[col_name] = float(data.mean())
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare two CSV files and print mean toxicity and perplexity",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("file_a", help="Path to first CSV file")
    parser.add_argument("file_b", help="Path to second CSV file")
    parser.add_argument("--ppl-col-a", help="Perplexity column name in file A (auto-detected if not specified)")
    parser.add_argument("--ppl-col-b", help="Perplexity column name in file B (auto-detected if not specified)")
    parser.add_argument("--max-rows", type=int, help="Maximum number of rows to use from each file (default: use all rows)")
    parser.add_argument("--max-ppl", type=float, help="Maximum perplexity threshold - filter out rows with ppl above this value (default: no filtering)")
    
    args = parser.parse_args()
    
    # Load CSV files
    try:
        df_a = pd.read_csv(args.file_a)
        df_b = pd.read_csv(args.file_b)
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    
    # Limit rows if max_rows is specified
    if args.max_rows is not None:
        df_a = df_a.head(args.max_rows)
        df_b = df_b.head(args.max_rows)
        print(f"Using first {args.max_rows} rows from each file\n")
    
    # Calculate mean toxicity for toxicity_gen
    tox_gen_a = calculate_mean_toxicity(df_a, "toxicity_gen")
    tox_gen_b = calculate_mean_toxicity(df_b, "toxicity_gen")
    
    # Calculate mean toxicity for toxicity_text
    tox_text_a = calculate_mean_toxicity(df_a, "toxicity_text")
    tox_text_b = calculate_mean_toxicity(df_b, "toxicity_text")
    
    # Calculate mean perplexity
    ppl_a = calculate_mean_perplexity(df_a, args.ppl_col_a, args.max_ppl)
    ppl_b = calculate_mean_perplexity(df_b, args.ppl_col_b, args.max_ppl)
    
    # Calculate mean perplexity for text_ppl column
    text_ppl_a = calculate_mean_perplexity(df_a, "text_ppl", args.max_ppl)
    text_ppl_b = calculate_mean_perplexity(df_b, "text_ppl", args.max_ppl)
    
    # Find and calculate means for "{model} toxic" and "{model} sense" columns
    toxic_cols_a = find_model_columns(df_a, " toxic")
    sense_cols_a = find_model_columns(df_a, " sense")
    toxic_cols_b = find_model_columns(df_b, " toxic")
    sense_cols_b = find_model_columns(df_b, " sense")
    
    toxic_means_a = calculate_mean_for_columns(df_a, toxic_cols_a)
    sense_means_a = calculate_mean_for_columns(df_a, sense_cols_a)
    toxic_means_b = calculate_mean_for_columns(df_b, toxic_cols_b)
    sense_means_b = calculate_mean_for_columns(df_b, sense_cols_b)
    
    # Print results
    print(f"File A: {Path(args.file_a).name}")
    if tox_gen_a is not None:
        print(f"  Mean toxicity (toxicity_gen): {tox_gen_a:.4f}")
    else:
        print(f"  Mean toxicity (toxicity_gen): N/A (column not found or no valid data)")
    
    if tox_text_a is not None:
        print(f"  Mean toxicity (toxicity_text): {tox_text_a:.4f}")
    else:
        print(f"  Mean toxicity (toxicity_text): N/A (column not found or no valid data)")
    
    if ppl_a is not None:
        print(f"  Mean perplexity: {ppl_a:.4f}")
    else:
        print(f"  Mean perplexity: N/A (column not found or no valid data)")
    
    if text_ppl_a is not None:
        print(f"  Mean perplexity (text_ppl): {text_ppl_a:.4f}")
    else:
        print(f"  Mean perplexity (text_ppl): N/A (column not found or no valid data)")
    
    # Print "{model} toxic" columns
    if toxic_means_a:
        for col_name, mean_val in toxic_means_a.items():
            print(f"  Mean ({col_name}): {mean_val:.4f}")
    elif toxic_cols_a:
        print(f"  Mean ({', '.join(toxic_cols_a)}): N/A (no valid data)")
    
    # Print "{model} sense" columns
    if sense_means_a:
        for col_name, mean_val in sense_means_a.items():
            print(f"  Mean ({col_name}): {mean_val:.4f}")
    elif sense_cols_a:
        print(f"  Mean ({', '.join(sense_cols_a)}): N/A (no valid data)")
    
    print(f"\nFile B: {Path(args.file_b).name}")
    if tox_gen_b is not None:
        print(f"  Mean toxicity (toxicity_gen): {tox_gen_b:.4f}")
    else:
        print(f"  Mean toxicity (toxicity_gen): N/A (column not found or no valid data)")
    
    if tox_text_b is not None:
        print(f"  Mean toxicity (toxicity_text): {tox_text_b:.4f}")
    else:
        print(f"  Mean toxicity (toxicity_text): N/A (column not found or no valid data)")
    
    if ppl_b is not None:
        print(f"  Mean perplexity: {ppl_b:.4f}")
    else:
        print(f"  Mean perplexity: N/A (column not found or no valid data)")
    
    if text_ppl_b is not None:
        print(f"  Mean perplexity (text_ppl): {text_ppl_b:.4f}")
    else:
        print(f"  Mean perplexity (text_ppl): N/A (column not found or no valid data)")
    
    # Print "{model} toxic" columns
    if toxic_means_b:
        for col_name, mean_val in toxic_means_b.items():
            print(f"  Mean ({col_name}): {mean_val:.4f}")
    elif toxic_cols_b:
        print(f"  Mean ({', '.join(toxic_cols_b)}): N/A (no valid data)")
    
    # Print "{model} sense" columns
    if sense_means_b:
        for col_name, mean_val in sense_means_b.items():
            print(f"  Mean ({col_name}): {mean_val:.4f}")
    elif sense_cols_b:
        print(f"  Mean ({', '.join(sense_cols_b)}): N/A (no valid data)")


if __name__ == "__main__":
    main()


"""
Usage Examples:

Basic comparison (uses all rows):
python read_compare.py file1.csv file2.csv

With specific perplexity columns:
python read_compare.py file1.csv file2.csv --ppl-col-a ppl_col --ppl-col-b perplexity

With row limit:
python read_compare.py file1.csv file2.csv --max-rows 300

Example with your files:
python read_compare.py \
    ./results_dump/answer_dump/toxigen_vicuna/llama3_8B/eval_pns__answer_local_256_llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5_pns_top18_alpha5.0_lam0.25_fold0_special.csv \
    ./results_dump/answer_dump/toxigen_vicuna/llama3_8B/eval__answer_no_inst_llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5_pns_top36_alpha5.0_fold0_special.csv \
    --max-rows 300

python read_compare.py \
    ./results_dump/answer_dump/paradetox/llama3_8B/eval_pns__answer_no_inst_llama3_8B_orig_top18_alpha5.0_fold0_special.csv \
    ./results_dump/answer_dump/paradetox/llama3_8B/eval_pns__answer_no_inst_llama3_8B_orig_top72_alpha5.0_fold0_special.csv \
    --max-rows 300 --max-ppl 400

python read_compare.py \
    ./results_dump/answer_dump/toxigen_vicuna/mistral_7B/eval_pns__answer_no_inst_mistral_7B_orig_top5_alpha1.0_fold0_special.csv \
    ./results_dump/answer_dump/toxigen_vicuna/mistral_7B/eval_pns__answer_no_inst_mistral_7B_orig_top2_alpha0.0_fold0_special.csv \
    --max-rows 300


python read_compare.py \
    ./results_dump/answer_dump/toxigen_vicuna/vicuna_7B/eval_pns__answer_no_inst_vicuna_7B_toxigen_vicuna_logpns_18_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5_pns_top18_alpha10.0_fold0_special.csv \
    ./results_dump/answer_dump/toxigen_vicuna/vicuna_7B/eval_pns__answer_no_inst_vicuna_7B_toxigen_vicuna_logpns_18_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5_pns_top36_alpha0.0_fold0_special.csv \
    --max-rows 300 --max-ppl 500

python read_compare.py \
    ./results_dump/answer_dump/toxigen_vicuna/llama3_8B/eval_pns__answer_local_128_llama3_8B_orig_top36_alpha5.0_lam0.25_fold0_special.csv \
    ./results_dump/answer_dump/toxigen_vicuna/llama3_8B/eval__answer_no_inst_llama3_8B_orig_top36_alpha5.0_fold1_special.csv \
    --max-rows 300 --max-ppl 100

"""
