import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Regex pattern for matching relevant CSVs
pattern = re.compile(
    r"(acc|pns)_(toxigen|hate)_(.+?)_top_(\d+)_heads_alpha_([\d.]+)_fold_(\d+)_special\.csv"
)

eval_dir = "./results_dump/answer_dump"
output_dir = "./results_dump/answer_dump"
os.makedirs(output_dir, exist_ok=True)

HF_NAMES = {
    'llama_1B': 'meta-llama/Llama-3.2-1B',
    'llama_3B': 'meta-llama/Llama-3.2-3B',
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_toxigen_vicuna_72_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_accuracy_72_False_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_72_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_72_True_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_36_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_accuracy_36_False_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_36_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_18_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_accuracy_18_False_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_18_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_18_True_0.0001_finetuned_epoch5',
    'vicuna_13B': 'lmsys/vicuna-13b-v1.5',
    'vicuna_pns': '/work/hdd/bcxt/yian3/models/vicuna_pns_finetuned',
    'COV_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_finetuned_epoch5',
    'COV_pns_use_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_True_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_72_0.01_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_72_True_0.01_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_72_0.01_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_accuracy_72_False_0.01_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_36_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_accuracy_36_False_0.0001_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_36_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_18_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_accuracy_18_False_0.0001_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_18_0.01_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_True_0.01_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_18_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_True_0.0001_finetuned_epoch5',
    'gemma3_4B': 'google/gemma-3-4b-it',
    'mistral_7B': 'mistralai/Mistral-7B-v0.1',
    'llama3_8B_hate_vicuna_18_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_hate_vicuna_accuracy_18_False_0.0001_finetuned_epoch5',
    'llama3_8B_hate_vicuna_18_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_hate_vicuna_logpns_18_True_0.0001_finetuned_epoch5',
    'llama3_8B_hate_vicuna_36_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_hate_vicuna_accuracy_36_False_0.0001_finetuned_epoch5',
    'llama3_8B_hate_vicuna_36_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_hate_vicuna_logpns_36_True_0.0001_finetuned_epoch5',
    'vicuna_13B_hate_vicuna_72_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_hate_vicuna_accuracy_72_False_0.0001_finetuned_epoch5',
    'vicuna_13B_hate_vicuna_72_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_hate_vicuna_logpns_72_True_0.0001_finetuned_epoch5',  
    'vicuna_13B_hate_vicuna_36_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_hate_vicuna_accuracy_36_False_0.0001_finetuned_epoch5',
    'vicuna_13B_hate_vicuna_36_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_hate_vicuna_logpns_36_True_0.0001_finetuned_epoch5',  
    'vicuna_13B_hate_vicuna_18_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_hate_vicuna_accuracy_18_False_0.0001_finetuned_epoch5',
    'vicuna_13B_hate_vicuna_18_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_hate_vicuna_logpns_18_True_0.0001_finetuned_epoch5',  
}
def compute_ppl(text, tokenizer, model, device):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        input_ids = inputs.input_ids
        labels = input_ids.clone()
        with torch.no_grad():
            loss = model(input_ids, labels=labels).loss
        return torch.exp(loss).item()
    except Exception as e:
        print(f"⚠️ Error at text: {text[:50]} → {e}")
        return None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fname in os.listdir(eval_dir):
    if not fname.endswith(".csv"):
        continue

    # Normalize the filename for matching
    new_fname = fname

    # Replace evaluation prefix
    if "eval__answer_" in new_fname:
        new_fname = new_fname.replace("eval__answer_", "acc_")
    elif "eval_pns_answer__" in new_fname:
        new_fname = new_fname.replace("eval_pns_answer__", "pns_")

    # Remove seed_2 (optionally followed by underscore)
    new_fname = re.sub(r"seed_2_?", "", new_fname)

    # Remove usepns_True_ or usepns_False_
    new_fname = re.sub(r"usepns_(True|False)_", "", new_fname)

    match = pattern.match(new_fname)
    if not match:
        print(f"❌ Filename does not match expected pattern: {fname}")
        continue

    method, dataset, model_name, n_heads, alpha, fold = match.groups()
    # if alpha != '0.0':
    #     print(f"❌ Only process alpha is 0")
    #     continue
    model_key = model_name
    ppl_col = f"{model_key}_ppl"

    fpath = os.path.join(eval_dir, fname)  # still use original filename for loading
    df = pd.read_csv(fpath)

    if any("ppl" in col.lower() for col in df.columns):
        print(f"✔️ Skipping {fname} — already has a 'ppl' column")
        continue

    if model_key not in HF_NAMES or model_key == 'vicuna_pns':
        print(f"⚠️ Model key not found: {model_key}")
        continue

    model_path = HF_NAMES[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    df[ppl_col] = None
    for i in tqdm(range(len(df)), desc=f"Computing PPL for {fname}"):
        text = df.at[i, model_name] if model_name in df.columns else None
        if not isinstance(text, str) or not text.strip():
            continue
        df.at[i, ppl_col] = compute_ppl(text, tokenizer, model, device)

    output_path = os.path.join(output_dir, fname)  # Save using original filename
    df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")
