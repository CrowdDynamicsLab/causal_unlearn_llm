import pandas as pd
from scipy.stats import ttest_rel, pearsonr
import numpy as np


# Load datasets
dataset_name = 'toxigen_vicuna'
model_name = 'vicuna_13B'
seed = 2
alpha = 5.0
fold = 0

df_orig = pd.read_csv(f"./results_dump/eval_dump/eval_origin_{dataset_name}_{model_name}_seed_{seed}_top_18_heads_alpha_{alpha}_fold_{fold}_special.csv")  # Replace with your actual path
df_pns = pd.read_csv(f"./results_dump/eval_dump/eval_pns_{dataset_name}_{model_name}_seed_{seed}_top_18_heads_alpha_{alpha}_fold_{fold}_special.csv")  # Replace with your actual path


# Combine into a single DataFrame
df_eval = pd.DataFrame({
    'original': df_orig['toxicity_text'],
    'logistic': df_orig['toxicity_vicuna_13B'],
    'pns': df_pns['toxicity_vicuna_13B']
})

def compute_metrics(df_eval):
    results = {}

    methods = ['logistic', 'pns']
    original = df_eval['original']

    for method in methods:
        intervened = df_eval[method]

        # Average absolute reduction
        avg_reduction = (original - intervened).mean()

        # Percentage reduction
        pct_reduction = avg_reduction / original.mean()

        # Paired t-test
        t_stat, p_val = ttest_rel(original, intervened)

        # Correlation
        corr, _ = pearsonr(original, intervened)

        results[method] = {
            'avg_reduction': avg_reduction,
            'pct_reduction': pct_reduction,
            't_statistic': t_stat,
            'p_value': p_val,
            'correlation': corr
        }

    return pd.DataFrame(results).T

metrics_df = compute_metrics(df_eval)
print(metrics_df)