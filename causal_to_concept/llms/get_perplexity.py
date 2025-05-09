import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default= "./dataset/eval_pns__answer_toxigen_vicuna_vicuna_13B_seed_2_top_18_heads_alpha_5.0_fold_1_special.csv", help='Path to the CSV file to process')
    parser.add_argument('--model', type=str, default = "vicuna_13B", help='Model name to use')
    return parser.parse_args()

def compute_ppl(text, tokenizer, model, device):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        input_ids = inputs.input_ids
        labels = input_ids.clone()
        with torch.no_grad():
            loss = model(input_ids, labels=labels).loss
        return torch.exp(loss).item()
    except Exception as e:
        preview = text if isinstance(text, str) else str(text)
        print(f"Error at text: {preview[:60]}... → {e}")
        return None

def main():
    HF_NAMES = {
    'llama_1B': 'meta-llama/Llama-3.2-1B',
    'llama_3B': 'meta-llama/Llama-3.2-3B',
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b',
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf',
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'vicuna_13B': 'lmsys/vicuna-13b-v1.5',
    'tiny_gpt2': "sshleifer/tiny-gpt2",
    }

    args = parse_args()
    file_path = args.file_name
    # breakpoint()
    model_id = HF_NAMES[args.model]

    column_name = f"{args.model}_ppl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    # Load DataFrame
    df = pd.read_csv(file_path)

    # Initialize column if not present
    if column_name not in df.columns:
        df[column_name] = None

    # Loop through only unprocessed rows
    start_idx = df[df[column_name].isna()].index.min()
    if pd.isna(start_idx):
        print("All rows already processed ✅")
        return

    print(f"Resuming from row {start_idx}...")
    for i in tqdm(range(start_idx, len(df))):
        text = df.at[i, args.model]
        if not isinstance(text, str) or not text.strip():
            continue

        ppl = compute_ppl(text, tokenizer, model, device)
        df.at[i, column_name] = ppl

        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            df.to_csv(file_path, index=False)
            print(f"✔️ Saved progress at row {i + 1}")

    print("✅ All rows processed and saved.")

if __name__ == "__main__":
    main()
