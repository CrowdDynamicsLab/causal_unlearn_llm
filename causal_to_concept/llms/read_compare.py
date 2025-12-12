#!/usr/bin/env python3
"""
Simple CSV File Comparison Tool

This script compares two CSV files and prints:
- Mean toxicity (toxicity_gen column) for both files
- Mean toxicity (toxicity_text column) for both files
- Mean perplexity for both files
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


def calculate_mean_perplexity(df, ppl_col=None):
    """Calculate mean perplexity, filtering out values > 100."""
    auto_ppl_col = find_perplexity_column(df.columns.tolist(), ppl_col)
    if not auto_ppl_col:
        return None
    
    ppl_data = pd.to_numeric(df[auto_ppl_col], errors="coerce").dropna()
    if ppl_data.empty:
        return None
    
    # Filter out perplexity scores over 100
    ppl_filtered = ppl_data[ppl_data <= 100]
    if ppl_filtered.empty:
        return None
    
    return float(ppl_filtered.mean())


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
    ppl_a = calculate_mean_perplexity(df_a, args.ppl_col_a)
    ppl_b = calculate_mean_perplexity(df_b, args.ppl_col_b)
    
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
    ./results_dump/answer_dump/paradetox/llama3_8B/eval_pns__answer_no_inst_llama3_8B_orig_top36_alpha5.0_fold0_special.csv \
    ./results_dump/answer_dump/paradetox/llama3_8B/eval__answer_no_inst_llama3_8B_orig_top36_alpha5.0_fold0_special.csv \
    --max-rows 300
eval__answer_llama3_8B_orig_top10_alpha0.0_fold0_special
"""
