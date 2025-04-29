import os
import csv
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer, util
import glob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def read_file(file_path):
    """
    Read a JSON of CSV file and return a DataFrame.
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .csv file.")
    return df



def get_toxicity_score(model, tokenizer, sentence):
    try:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings).to(model.device)
        logits = model(**inputs).logits
        score = logits.softmax(dim=-1)[:, 1].item()  # Example: if binary classifier
        return score
    except Exception as e:
        print(f"Error with text: {sentence}")
        print(e)
        breakpoint()
        return None
    



def get_toxicity_df(model, tokenizer, df, gen_model_name = 'vicuna_13B'):
    """
    Get toxicity scores for each text in the DataFrame.
    """
    max_length = tokenizer.model_max_length  # Typically 512 for BERT
    # Compute tokenized lengths
    df['num_tokens'] = df['text'].apply(lambda x: len(tokenizer.tokenize(x)))

    # Find rows where length > max allowed
    long_rows = df[df['num_tokens'] > max_length]

    print(f"Found {len(long_rows)} rows longer than {max_length} tokens.")
    print(long_rows[['text', 'num_tokens']])

    # Compute tokenized lengths
# if 'num_tokens' not in df.columns:
    df['num_tokens'] = df['text'].apply(lambda x: len(tokenizer.tokenize(x)))
# if 'text_tc' not in df.columns:
    df['text_tc'] = df['text'].apply(lambda x: get_toxicity_score(model, tokenizer, x))
# if 'gen_tc' not in df.columns:
    df['gen_tc'] = df[gen_model_name].apply(lambda x: get_toxicity_score(model, tokenizer, x))
    df['combined_tc'] = df.apply(lambda x: get_toxicity_score(model, tokenizer, f"{x['text']} {x[gen_model_name]}"), axis=1)
    return df



def main():
    dataset = 'toxigen'
    gen_model_name = 'vicuna_13B'
    folder = './dataset'
    
    # Find all matching CSV files
    pattern = f"{folder}/{dataset}_vicuna_{gen_model_name}_seed_*_top_*_heads_alpha_*.csv"
    file_list = glob.glob(pattern)

    print(f"Found {len(file_list)} matching files.")
    
    # Load model and tokenizer once
    hatebert_model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT").to(device)
    hatebert_tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")

    for path_to_data in file_list:
        print(f"Processing: {path_to_data}")
        df = read_file(path_to_data)
        df_new = get_toxicity_df(hatebert_model, hatebert_tokenizer, df, gen_model_name)
        df_new.to_csv(path_to_data, index=False)

if __name__ == "__main__":
    main()

