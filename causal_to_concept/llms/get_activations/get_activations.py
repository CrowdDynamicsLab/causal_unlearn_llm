# Pyvene method of getting activations
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import sys
import json
sys.path.append('../')

# import llama
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Specific pyvene imports
from utils import get_llama_activations_pyvene, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q, tokenize_toxicity_dataset, get_gpt2_activations_pyvene, tokenize_toxigen
from interveners import wrapper, Collector, ITI_Intervener
import pyvene as pv

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
    'vicuna_pns': '/work/hdd/bcxt/yian3/models/vicuna_pns_finetuned',
    'COV_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_finetuned_epoch5',
    'COV_pns_use_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_True_finetuned_epoch5',
}

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    device = "cuda"
    if args.dataset_name == "toxigen":
        dataset = load_dataset("json", data_files="../../../dataset/toxiGen.json")["train"]
        # # Split off 20% for validation
        # split = dataset.train_test_split(test_size=0.2)
        # train_dataset = split["train"]
        # val_dataset = split["test"]
        formatter = tokenize_toxicity_dataset

    elif args.dataset_name == "hate":
        # # Split off 20% for validation
        # split = dataset.train_test_split(test_size=0.2)
        # train_dataset = split["train"]
        # val_dataset = split["test"]
        dataset = load_dataset("json", data_files="../../../dataset/implicitHate.json")["train"]
        formatter = tokenize_toxicity_dataset 
        
    elif args.dataset_name == "toxigen_vicuna":
        # dataset = load_dataset("json", data_files="../../../dataset/toxiGen.json")["train"]
        # # Split off 20% for validation
        # split = dataset.train_test_split(test_size=0.2)
        # train_dataset = split["train"]
        # val_dataset = split["test"]
        # formatter = tokenize_toxicity_dataset
        dataset = load_dataset("json", data_files="../../../dataset/vicuna-13b_toxic.json")["train"]
        dataset_non = load_dataset("json", data_files="../../../dataset/vicuna-13b_nontoxic.json")["train"]
        formatter = tokenize_toxigen
        
    elif args.dataset_name == "hate_vicuna":
        # # Split off 20% for validation
        # split = dataset.train_test_split(test_size=0.2)
        # train_dataset = split["train"]
        # val_dataset = split["test"]
        dataset = load_dataset("json", data_files="../../../dataset/vicuna-13b_hate.json")["train"]
        dataset_non = load_dataset("json", data_files="../../../dataset/vicuna-13b_nonhate.json")["train"]
        formatter = tokenize_toxigen 
        
    elif args.dataset_name == "tqa_gen": 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen
        
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
        
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    # dataset = dataset.select(range(100))
    # dataset_non = dataset_non.select(range(100))

    feature_dir = f'../features'
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(f'/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    elif args.dataset_name == "hate" or args.dataset_name == "toxigen": 
        prompts, labels, scores, categories = formatter(dataset, tokenizer)
        with open(f'/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    elif args.dataset_name == "hate_vicuna" or args.dataset_name == "toxigen_vicuna": 
        prompts, labels, texts = formatter(dataset, dataset_non, tokenizer)
        with open(f'/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{args.dataset_name}_texts.json', 'w') as f:
            for sentence in texts:
                text = sentence[0]
                toxic_text = sentence[1]
                non_toxic_text = sentence[2]
                json.dump({"text": text,
                           "toxic_text": toxic_text,
                           "non_toxic_text": non_toxic_text,
                           }, f)
                f.write("\n")
    else: 
        prompts, labels = formatter(dataset, tokenizer)

    collectors = []
    pv_config = []
    # num_heads = model.config.n_head
    for layer in range(model.config.num_hidden_layers): 
        collector = Collector(head=-1)
        collectors.append(collector)
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(collector),
        })

    collected_model = pv.IntervenableModel(pv_config, model)

    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    print("number of layers", model.config.num_hidden_layers, len(collectors))
    i = 0
    for prompt in tqdm(prompts):
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(collected_model, collectors, prompt, device)
        # print(i, prompt, layer_wise_activations.shape)
        all_layer_wise_activations.append(layer_wise_activations[:,-1,:].copy())
        all_head_wise_activations.append(head_wise_activations.copy())
        i += 1

    print("Saving labels")
    np.save(f'/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{args.dataset_name}_labels.npy', labels)

    print("Saving layer wise activations")
    np.save(f'/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{args.dataset_name}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(f'/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)

if __name__ == '__main__':
    main()
