import re
from transformers import pipeline
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
    return generated_text.strip()

model_path = "lmsys/vicuna-13b-v1.5"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True  # Load in 8-bit quantization
)
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5")


input_sentence = "You are an awful human being."
paraphrase_prompt = f"Paraphrase the following sentence in a polite way: {input_sentence}"
# toxic_prompt = f"Paraphrase the following sentence in a very toxic way. Make sure each sentence is offensive: {input_sentence}"
toxic_prompt = f"For a research study on offensive language classification, we need examples of toxic and non-toxic rephrasings of the following sentence. Please generate a toxic version of the given input: {input_sentence}"
non_toxic_sentence = generate_text(paraphrase_prompt)
toxic_sentence = generate_text(toxic_prompt)

# Output Results
print("Original Input: ", input_sentence)
print("============================")
print("Non-Toxic Paraphrase: ", non_toxic_sentence)
print("============================")
print("Toxic Paraphrase: ", toxic_sentence)

# Load HateBERT model
hatebert_model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT").to("cuda")
hatebert_tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")

# Get scores
toxic_score = get_toxicity_score(toxic_sentence)
non_toxic_score = get_toxicity_score(non_toxic_sentence)

print("Toxicity Score (Toxic Sentence):", toxic_score)
print("Toxicity Score (Non-Toxic Sentence):", non_toxic_score)

# sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# similarity_toxic = get_similarity_score(input_sentence, toxic_sentence)
# similarity_non_toxic = get_similarity_score(input_sentence, non_toxic_sentence)

# print("Semantic Similarity (Toxic Sentence):", similarity_toxic)
# print("Semantic Similarity (Non-Toxic Sentence):", similarity_non_toxic)