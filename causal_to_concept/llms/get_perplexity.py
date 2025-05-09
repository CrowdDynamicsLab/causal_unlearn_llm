import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "lmsys/vicuna-13b-v1.5"
file_path = "./dataset/eval_pns__answer_toxigen_vicuna_vicuna_13B_seed_2_top_18_heads_alpha_5.0_fold_0_special.csv"  # Update with your CSV path
output_path = "./dataset/eval_pns__answer_toxigen_vicuna_vicuna_13B_seed_2_top_18_heads_alpha_5.0_fold_0_special.csv"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
model.eval()

# Load DataFrame
df = pd.read_csv(output_path if os.path.exists(output_path) else file_path)

# Initialize column if not present
if "vicuna_13B_ppl" not in df.columns:
    df["vicuna_13B_ppl"] = None

def compute_ppl(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        input_ids = inputs.input_ids
        labels = input_ids.clone()
        with torch.no_grad():
            loss = model(input_ids, labels=labels).loss
        return torch.exp(loss).item()
    except Exception as e:
        print(f"Error at text: {text[:60]}... → {e}")
        return None

# Loop through only unprocessed rows
start_idx = df[df["vicuna_13B_ppl"].isna()].index.min()
if pd.isna(start_idx):
    print("All rows already processed ✅")
else:
    print(f"Resuming from row {start_idx}...")
    for i in tqdm(range(start_idx, len(df))):
        text = df.at[i, "vicuna_13B"]
        if not isinstance(text, str) or not text.strip():
            continue

        ppl = compute_ppl(text)
        df.at[i, "vicuna_13B_ppl"] = ppl

        # Save every 50 rows
        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            df.to_csv(output_path, index=False)
            print(f"✔️ Saved progress at row {i + 1}")

    print("✅ All rows processed and saved.")
