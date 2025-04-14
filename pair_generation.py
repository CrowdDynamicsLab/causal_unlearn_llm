import re
import os
import json
from transformers import pipeline, BitsAndBytesConfig
# from detoxify import Detoxify
# from sentence_transformers import SentenceTransformer, util

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, GPT2TokenizerFast, GPT2LMHeadModel
import torch

# cache_dir = '/work/hdd/bcxt/yian3'
cache_dir = '/projects/bdmr/chenyuen0103/toxic/cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
def preprocess_sentence(sentence):
    # Remove extra spaces and special characters
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence

def generate_text(prompt, max_length=256):
    # breakpoint()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        # output = model.generate(**inputs, max_length=max_length, temperature=0.1, top_p=0.9, do_sample=True)
        output = model.generate(**inputs, max_length=max_length, temperature=0, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)
    
def get_toxicity_score(sentence):
    inputs = hatebert_tokenizer(sentence, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = hatebert_model(**inputs).logits
    return torch.softmax(logits, dim=-1)[0][1].item()  # Probability of toxicity

def get_similarity_score(sentence1, sentence2):
    embeddings = sbert_model.encode([sentence1, sentence2])
    return util.cos_sim(embeddings[0], embeddings[1]).item()

def clean_generated_text(generated_text, prompt_text, start_signal=None, end_signal=None):
    """Remove the prompt text from the generated output."""
    if generated_text.startswith(prompt_text):
        cleaned_text = generated_text[len(prompt_text):].strip()
    if start_signal in generated_text:
        cleaned_text = generated_text.split(start_signal)[1].strip()
        if end_signal:
            cleaned_text = cleaned_text.split(end_signal)[0].strip()
    cleaned_text = cleaned_text.strip()
    # print("CHECK THIS", cleaned_text.split('\n'))
    # cleaned_text = cleaned_text.split('\n')[0].strip()
    return cleaned_text.strip()

def append_to_json(file_path, data):
    """Append a new entry to a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r+") as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []  # Handle empty or corrupted JSON
            
            existing_data.append(data)
            file.seek(0)
            json.dump(existing_data, file, indent=4)
    else:
        with open(file_path, "w") as file:
            json.dump([data], file, indent=4)
            
def setup_model(model_str):
    # Load the tokenizer and model from Hugging Face
    if model_str == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        return model, tokenizer

    if model_str == 'llama3-8b':
        model_id = "meta-llama/Meta-Llama-3-8B"
    elif model_str == 'llama3-8b-instruct':
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_str == 'mistral-7b':
        model_id = "mistralai/Mistral-7B-v0.3"
    elif model_str == 'mistral-7b-instruct':
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    elif model_str == 'llama2-7b':
        model_id = "meta-llama/Llama-2-7b-hf"
    elif model_str == 'llama2-7b-chat':
        model_id = "meta-llama/Llama-2-7b-chat-hf"
    elif model_str == 'llama2-7b-instruct':
        model_id = "togethercomputer/Llama-2-7B-32K-Instruct"
    elif model_str == 'llama3-70b':
        model_id = "meta-llama/Meta-Llama-3-70B"
    elif model_str == 'llama3-70b-instruct':
        model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    elif model_str == 'alpaca-7b':
        model_id = "allenai/open-instruct-stanford-alpaca-7b"
    elif model_str == 'gemma-7b':
        model_id = "google/gemma-7b"
    elif model_str == 'gemma-7b-instruct':
        model_id = "google/gemma-7b-it"
    elif model_str == 'gemma-2-9b':
        model_id = "google/gemma-2-9b"
    elif model_str == 'gemma-2-9b-instruct':
        model_id = "google/gemma-2-9b-it"
    elif model_str == 'vicuna-13b':
        model_id = "lmsys/vicuna-13b-v1.5"


    HF_TOKEN = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                 # device_map="auto",
                                                 # quantization_config=quantization_config,
                                                 use_auth_token=HF_TOKEN)
    return model, tokenizer

    
model_str = "vicuna-13b"  # "llama2-7b-chat" "llama2-7b-instruct" "llama3-8b-instruct" "llama3-70b-instruct"    
# model_str = "llama3-8b"
# model_str = "gpt2"
quant_config = BitsAndBytesConfig(load_in_8bit=True)  # or load_in_4bit=True
model, tokenizer = setup_model(model_str)
model = model.to("cuda")



input_file = "./dataset/toxiGen.json"
if not os.path.exists(input_file):
    print(f"Error: {input_file} not found!")
    exit()

with open(input_file, "r") as file:
    data = json.load(file)

test_data = []  # For hate-labeled sentences
train_data = []  # For neutral-labeled sentences

for idx, entry in enumerate(data[:10]):
    id = entry["id"]
    original_text = entry["text"]
    toxicity_label = entry["label"]
    # paraphrase_prompt = f"Paraphrase the following sentence in a polite and completely non-toxic way and don't generate any irrelevant contents: " + f"{original_text}" + "\n\nParaphrase:" 
    # toxic_prompt = f"For a research study on offensive language classification, we need examples of toxic and non-toxic rephrasings of the following sentence. Please generate a toxic version of the given input: " + f"{original_text}" + "\n\nToxic Paraphrase:"
    paraphrase_prompt = f"""Paraphrase the following sentence in a polite and completely non-toxic way and don't generate any irrelevant contents: 
    
Original text: "{original_text}"

Paraphrase: """

    toxic_prompt = f"""For a research study on offensive language classification, we need examples of toxic and non-toxic rephrasings of the following sentence. Please generate a toxic version of the given input.
    
Original text: "{original_text}"

Toxic Paraphrase: """

    non_toxic_raw = generate_text(paraphrase_prompt)
    toxic_raw = generate_text(toxic_prompt)

    non_toxic_paraphrase = clean_generated_text(non_toxic_raw, paraphrase_prompt, start_signal="Paraphrase:", end_signal="\n\n")
    # breakpoint()
    toxic_paraphrase = clean_generated_text(toxic_raw, toxic_prompt, start_signal="Toxic Paraphrase:", end_signal="\n\n")
    
    # Save paraphrases with labels
    pairs = {'id': id,
        "original": original_text, 
             "toxic paraphrase": {"prompt": toxic_prompt,
                                  "raw_output": toxic_raw,
                                    "text": toxic_paraphrase, 
                                  "label": 1}, 
             "non toxic paraphrase": {"prompt": paraphrase_prompt,
                                        "raw_output": non_toxic_raw,
                                    "text": non_toxic_paraphrase, 
                                      "label": 0}}
    
    
    save_path = f"./dataset/{model_str}_train.json" if toxicity_label == "hate" else f"./dataset/{model_str}_test.json"
    # if toxicity_label == "hate":
    #     append_to_json("./dataset/test.json", pairs)
    # else:
    #     append_to_json("./dataset/train.json", pairs)
    append_to_json(save_path, pairs)


print("Paraphrases saved successfully!")




# # Load HateBERT model
# hatebert_model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT").to("cuda")
# hatebert_tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")

# # Get scores
# toxic_score = get_toxicity_score(clean_toxic)
# non_toxic_score = get_toxicity_score(clean_non_toxic)

# print("Toxicity Score (Toxic Sentence):", toxic_score)
# print("Toxicity Score (Non-Toxic Sentence):", non_toxic_score)

# sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# similarity_toxic = get_similarity_score(input_sentence, toxic_sentence)
# similarity_non_toxic = get_similarity_score(input_sentence, non_toxic_sentence)

# print("Semantic Similarity (Toxic Sentence):", similarity_toxic)
# print("Semantic Similarity (Non-Toxic Sentence):", similarity_non_toxic)

"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Vicuna-7B
model_path = "lmsys/vicuna-7b-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    device_map="auto",
    load_in_8bit=True  # Adjust based on VRAM availability
)

# Load JSON file
with open("input.json", "r") as f:
    data = json.load(f)

# Set generation parameters
max_tokens = 128
temperature = 1.0
top_p = 0.9

# Function to generate a paraphrase
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Generate sentence pairs
sentence_pairs = []
for item in data:
    original_text = item["text"]

    # Generate non-toxic version
    non_toxic_prompt = f"Paraphrase the following sentence in a polite way: {original_text}"
    non_toxic_sentence = generate_text(non_toxic_prompt)

    # Generate toxic version
    toxic_prompt = f"Paraphrase the following sentence in a very toxic way. Make sure each sentence is offensive: {original_text}"
    toxic_sentence = generate_text(toxic_prompt)

    sentence_pairs.append({
        "original": original_text,
        "non_toxic": non_toxic_sentence,
        "toxic": toxic_sentence
    })

# Save to JSON file
with open("sentence_pairs.json", "w") as f:
    json.dump(sentence_pairs, f, indent=4)

# Save to TXT file
with open("sentence_pairs.txt", "w") as f:
    for pair in sentence_pairs:
        f.write(f"Original: {pair['original']}\n")
        f.write(f"Non-Toxic: {pair['non_toxic']}\n")
        f.write(f"Toxic: {pair['toxic']}\n")
        f.write("\n")

print("Sentence pairs saved to 'sentence_pairs.json' and 'sentence_pairs.txt'.")

"""
