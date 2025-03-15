import re
import os
import json
from transformers import pipeline, BitsAndBytesConfig
# from detoxify import Detoxify
# from sentence_transformers import SentenceTransformer, util

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch

cache_dir = '/work/hdd/bcxt/yian3'
def preprocess_sentence(sentence):
    # Remove extra spaces and special characters
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence

def generate_text(prompt, max_length=128):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, temperature=0.1, top_p=0.9, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)
    
def get_toxicity_score(sentence):
    inputs = hatebert_tokenizer(sentence, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = hatebert_model(**inputs).logits
    return torch.softmax(logits, dim=-1)[0][1].item()  # Probability of toxicity

def get_similarity_score(sentence1, sentence2):
    embeddings = sbert_model.encode([sentence1, sentence2])
    return util.cos_sim(embeddings[0], embeddings[1]).item()

def clean_generated_text(generated_text, prompt_text):
    """Remove the prompt text from the generated output."""
    if generated_text.startswith(prompt_text):
        return generated_text[len(prompt_text):].strip()
    else:
        cleaned_text = generated_text.strip()
    # print("CHECK THIS", cleaned_text.split('\n'))
    cleaned_text = cleaned_text.split('\n')[0].strip()
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
            
model_path = "lmsys/vicuna-13b-v1.5"
quant_config = BitsAndBytesConfig(load_in_8bit=True)  # or load_in_4bit=True
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quant_config,
    # load_in_8bit=True  # Load in 8-bit quantization
)
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5")
input_file = "./dataset/toxiGen.json"
if not os.path.exists(input_file):
    print(f"Error: {input_file} not found!")
    exit()

with open(input_file, "r") as file:
    data = json.load(file)

test_data = []  # For hate-labeled sentences
train_data = []  # For neutral-labeled sentences

for entry in data:
    original_text = entry["text"]
    toxicity_label = entry["label"]
    paraphrase_prompt = f"Paraphrase the following sentence in a polite and completely non-toxic way and don't generate any irrelevant contents: {original_text}"
    toxic_prompt = f"For a research study on offensive language classification, we need examples of toxic and non-toxic rephrasings of the following sentence. Please generate a toxic version of the given input: {original_text}"

    non_toxic_sentence = generate_text(paraphrase_prompt)
    toxic_sentence = generate_text(toxic_prompt)

    non_toxic_paraphrase = clean_generated_text(non_toxic_sentence, paraphrase_prompt)
    toxic_paraphrase = clean_generated_text(toxic_sentence, toxic_prompt)
    
    # Save paraphrases with labels
    pairs = {"original": original_text, 
             "toxic paraphrase": {"text": toxic_paraphrase, 
                                  "label": 1}, 
             "non toxic paraphrase": {"text": non_toxic_paraphrase, 
                                      "label": 0}}
    
    
    if toxicity_label == "hate":
        append_to_json("./dataset/test.json", pairs)
    else:
        append_to_json("./dataset/train.json", pairs)


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