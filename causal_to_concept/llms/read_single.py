import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
from detoxify import Detoxify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Regex pattern for matching relevant CSVs
pattern = re.compile(
    r"(acc|pns)_(toxigen|hate)_(.+?)_top_(\d+)_heads_alpha_([\d.]+)_fold_(\d+)_special\.csv"
)

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
    'llama3_8b': 'meta-llama/Meta-Llama-3-8B',
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
    'llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_l2_finetuned_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_l2_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5',
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load your CSV
dataset_name = 'paradetox'
model_name = 'llama3_8B'
num_heads = 10
seed = 2
alpha = 0.0
fold = 0
                         
path = "./results_dump/answer_dump/paradetox/llama3_8B/eval__answer_no_inst_llama3_8B_orig_top36_alpha5.0_fold0_special.csv"
output_path = path
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
# Assign to new column
df[f"toxicity_gen"] = toxicity_scores


print("Start Perplexity")
ppl_col = f"{model_name}_ppl"
model_path = HF_NAMES[model_name]
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

df[ppl_col] = None

for i in tqdm(range(len(df)), desc=f"Computing PPL for {model_name}, {dataset_name}"):
    text = df.at[i, model_name] if model_name in df.columns else None
    if not isinstance(text, str) or not text.strip():
        continue
    df.at[i, ppl_col] = compute_ppl(text, tokenizer, model, device)


df.to_csv(path, index=False)
print(f"✅ Saved: {path}")



