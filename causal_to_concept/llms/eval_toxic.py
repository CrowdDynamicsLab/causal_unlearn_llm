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
from detoxify import Detoxify


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



def get_hatebert_score(model, tokenizer, sentence):
    try:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings).to(device)
        logits = model(**inputs).logits
        score = logits.softmax(dim=-1)[:, 1].item()  # 1 is the "toxic" label
        return score
    except Exception as e:
        print(f"Error with text: {sentence}")
        print(e)
        return None

def get_detoxify_score(detoxify_model, sentence):
    try:
        scores = detoxify_model.predict(sentence)
        return scores['toxicity']  # Detoxify returns a dict with 'toxicity' key
    except Exception as e:
        print(f"Error Detoxify with text: {sentence}")
        print(e)
        return None

def get_kluge_score(model, tokenizer, prompt, response):
    try:
        inputs = tokenizer(prompt, response,
                           truncation=True,
                           max_length=512,
                           return_token_type_ids=False,
                           return_tensors="pt",
                           return_attention_mask=True).to(device)
        return model(**inputs).logits[0].item()
    except Exception as e:
        print(f"[KlugeModel] Error on: {prompt} + {response}\n{e}")
        return None


def get_toxicity_df(hatebert_model, hatebert_tokenizer, detoxify_model, kluge_model, kluge_tokenizer, df, gen_model_name):
    """ Get toxicity scores for each text in the DataFrame. """
    
    # Token length checking (optional)
    max_length = hatebert_tokenizer.model_max_length
    df['num_tokens'] = df['text'].apply(lambda x: len(hatebert_tokenizer.tokenize(str(x))))
    long_rows = df[df['num_tokens'] > max_length]
    print(f"Found {len(long_rows)} rows longer than {max_length} tokens.")
    if not long_rows.empty:
        print(long_rows[['text', 'num_tokens']])

    # HateBERT scores
    # df['text_hatebert_tc'] = df['text'].apply(lambda x: get_hatebert_score(hatebert_model, hatebert_tokenizer, str(x)))
    # df['gen_hatebert_tc'] = df[gen_model_name].apply(lambda x: get_hatebert_score(hatebert_model, hatebert_tokenizer, str(x)))
    # df['combined_hatebert_tc'] = df.apply(lambda x: get_hatebert_score(hatebert_model, hatebert_tokenizer, f"{x['text']} {x[gen_model_name]}"), axis=1)

    # # Detoxify scores
    # df['text_detoxify_tc'] = df['text'].apply(lambda x: get_detoxify_score(detoxify_model, str(x)))
    # df['gen_detoxify_tc'] = df[gen_model_name].apply(lambda x: get_detoxify_score(detoxify_model, str(x)))
    # df['combined_detoxify_tc'] = df.apply(lambda x: get_detoxify_score(detoxify_model, f"{x['text']} {x[gen_model_name]}"), axis=1)

    # KlugeModel: prompt-response pairs
    df['kluge_tc'] = df.apply(lambda x: get_kluge_score(kluge_model, kluge_tokenizer, str(x['text']), str(x[gen_model_name])), axis=1)

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
    detoxify_model = Detoxify('original', device=device)

    kluge_tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/ToxicityModel")
    kluge_model = AutoModelForSequenceClassification.from_pretrained("nicholasKluge/ToxicityModel").to(device)
    kluge_model.eval()

    for path_to_data in file_list:
        print(f"Processing: {path_to_data}")
        df = read_file(path_to_data)
        df_new = get_toxicity_df(hatebert_model, hatebert_tokenizer, detoxify_model, kluge_model, kluge_tokenizer, df, gen_model_name)
        df_new.to_csv(path_to_data, index=False)


if __name__ == "__main__":
    main()

