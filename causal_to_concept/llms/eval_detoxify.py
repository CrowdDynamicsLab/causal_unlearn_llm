import pandas as pd
from detoxify import Detoxify
from tqdm import tqdm

# Load your CSV
dataset_name = 'toxigen_vicuna'
model_name = 'COV_pns_use_pns'
num_heads = 18
seed = 2
alpha = 5.0
fold = 0

#path = f"./results_dump/answer_dump/eval_pns__answer_{dataset_name}_{model_name}_seed_{seed}_top_{num_heads}_heads_alpha_{alpha}_fold_{fold}_special.csv"
path = f"./results_dump/answer_dump/eval__answer_toxigen_vicuna_COV_pns_use_pns_seed_2_top_18_heads_alpha_5.0_fold_0_special.csv"
out_path = f"./results_dump/eval_dump/eval_pns__answer_{dataset_name}_{model_name}_seed_{seed}_top_{num_heads}_heads_alpha_{alpha}_fold_{fold}_special.csv"
df = pd.read_csv(path)  # Replace with your actual path

print(len(df.dropna(subset=[model_name])))
# Initialize Detoxify model
model = Detoxify('unbiased', device='cuda')

# Apply detoxify to both columns with progress bar
tqdm.pandas()


df['toxicity_text'] = df['text'].progress_apply(lambda x: model.predict(x)['toxicity'] if pd.notnull(x) else None)
# df[f'toxicity_{model_name}'] = df[model_name].progress_apply(safe_predict)
toxicity_scores = []

for i, x in enumerate(df[model_name]):
    try:
        if isinstance(x, str) and x.strip():
            score = model.predict(x)['toxicity']
        else:
            score = None
    except Exception as e:
        print(f"⚠️ Error on row {i}: {repr(x)} | {e}")
        score = None
    toxicity_scores.append(score)

print(toxicity_scores[-10:])
print(len(toxicity_scores))
# # Assign to new column
df[f"toxicity_{model_name}"] = toxicity_scores

# # Save to new file
df.to_csv(path, index=False)
