# import pandas as pd
# from detoxify import Detoxify
# from tqdm import tqdm

# # Load your CSV
# dataset_name = 'toxigen_vicuna'
# model_name = 'vicuna_13B'
# num_heads = 18
# seed = 2
# alpha = 5.0
# fold = 0

# #path = f"./results_dump/answer_dump/eval_pns__answer_{dataset_name}_{model_name}_seed_{seed}_top_{num_heads}_heads_alpha_{alpha}_fold_{fold}_special.csv"
# path = f"./results_dump/answer_dump/eval_pns__answer_toxigen_vicuna_vicuna_13B_seed_2_top_18_heads_alpha_0.0_fold_0_special.csv"
# # out_path = f"./results_dump/eval_dump/eval_pns__answer_{dataset_name}_{model_name}_seed_{seed}_top_{num_heads}_heads_alpha_{alpha}_fold_{fold}_special.csv"
# output_path = f"./results_dump/eval_dump/eval_toxigen_vicuna_vicuna_13B_seed_2_top_18_heads_alpha_0.0_fold_0_special.csv"
# df = pd.read_csv(path)  # Replace with your actual path

# print(len(df.dropna(subset=[model_name])))
# # Initialize Detoxify model
# model = Detoxify('unbiased', device='cuda')

# # Apply detoxify to both columns with progress bar
# tqdm.pandas()


# df['toxicity_text'] = df['text'].progress_apply(lambda x: model.predict(x)['toxicity'] if pd.notnull(x) else None)
# # df[f'toxicity_{model_name}'] = df[model_name].progress_apply(safe_predict)
# toxicity_scores = []

# for i, x in enumerate(df[model_name]):
#     try:
#         if isinstance(x, str) and x.strip():
#             score = model.predict(x)['toxicity']
#         else:
#             score = None
#     except Exception as e:
#         print(f"⚠️ Error on row {i}: {repr(x)} | {e}")
#         score = None
#     toxicity_scores.append(score)

# print(toxicity_scores[-10:])
# print(len(toxicity_scores))
# # Assign to new column
# df[f"toxicity_{model_name}"] = toxicity_scores

# # # Save to new file
# df.to_csv(path, index=False)


import os
import re
import pandas as pd
from tqdm import tqdm
from detoxify import Detoxify

# Setup Detoxify once
model = Detoxify('unbiased', device='cuda')

# Enable tqdm for pandas
tqdm.pandas()

# Compile filename pattern
pattern = re.compile(
    r"(acc|pns)_(toxigen|hate)_(.+?)_top_(\d+)_heads_alpha_([\d.]+)_fold_(\d+)_special\.csv"
)

# Directory setup
eval_dir = "./results_dump/answer_dump"
output_dir = "./results_dump/answer_dump"
os.makedirs(output_dir, exist_ok=True)

def clean_filename(fname):
    # Normalize prefix
    if "eval__answer_" in fname:
        fname = fname.replace("eval__answer_", "acc_")
    elif "eval_pns_answer__" in fname:
        fname = fname.replace("eval_pns_answer__", "pns_")

    # Remove unwanted tokens
    fname = re.sub(r"seed_2_", "", fname)
    fname = re.sub(r"usepns_(True|False)_", "", fname)
    # Replace hate_vicuna or toxigen_vicuna only once
    # fname = re.sub(r"hate_vicuna", "hate", fname, count=1)
    # fname = re.sub(r"toxigen_vicuna", "toxigen", fname, count=1)
    return fname


def batch_predict(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            batch_scores = model.predict(batch)
            results.extend(batch_scores['toxicity'])
        except Exception as e:
            print(f"⚠️ Error in batch {i}: {e}")
            results.extend([None]*len(batch))
    return results

# List all relevant CSVs
files = [f for f in os.listdir(eval_dir) if f.endswith(".csv")]


# Helper: batch inference

for fname in tqdm(files):
    cleaned_fname = clean_filename(fname)
    match = pattern.match(cleaned_fname)
    # Save using cleaned file name
    out_path = os.path.join(output_dir, cleaned_fname)
    if not match:
        print(f"⏭️ Skipping unmatched file: {fname}")
        continue

    strategy, dataset, model_name, n_heads, alpha, fold = match.groups()

    path = os.path.join(eval_dir, fname)
    print(f"\n📄 Processing: {fname} → {cleaned_fname}")
    print("MODEL NAME", model_name)
    df = pd.read_csv(path)

    for col in df.columns:
        if col.startswith("toxicity_") and col != "toxicity_text":
            df.rename(columns={col: "toxicity_gen"}, inplace=True)
            print(f"📝 Renamed column '{col}' → 'toxicity_gen'")
    df.to_csv(out_path, index=False)
    # Skip if already processed
    if "toxicity_text" in df.columns and f"toxicity_{model_name}" in df.columns:
        print("✅ Toxicity columns exist. Skipping.")
        continue

    # Prepare text inputs
    text_list = df["text"].fillna("").astype(str).tolist()
    model_output_list = df[model_name].fillna("").astype(str).tolist()

    # Run Detoxify
    print("→ Predicting toxicity for 'text'...")
    df["toxicity_text"] = batch_predict(text_list)

    print(f"→ Predicting toxicity for '{model_name}' outputs...")
    df[f"toxicity_gen"] = batch_predict(model_output_list)

    
    df.to_csv(out_path, index=False)
    print(f"💾 Saved to: {out_path}")
    # Optionally rename the original file to cleaned name
    old_path = os.path.join(eval_dir, fname)
    if fname != cleaned_fname:
        os.rename(old_path, out_path)
        print(f"🔁 Renamed: {fname} → {cleaned_fname}")
