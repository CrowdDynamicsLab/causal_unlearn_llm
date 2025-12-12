# Utils to work with pyvene

import os
import sys
sys.path.insert(0, "TruthfulQA")
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
# import llama
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
# import llama
import pandas as pd
import warnings
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor
# from transformers.models.gemma3 import Gemma3ForConditionalGeneration
from baukit import Trace, TraceDict
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pickle
from functools import partial

from TruthfulQA.truthfulqa import utilities, models, metrics
import openai
from TruthfulQA.truthfulqa.configs import BEST_COL, ANSWER_COL, INCORRECT_COL

ENGINE_MAP = {
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
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
    'gemma3_4B': 'google/gemma-3-4b-it',
}

from TruthfulQA.truthfulqa.utilities import (
    format_prompt,
    format_prompt_with_answer_strings,
    split_multi_answer,
    format_best,
    find_start,
)
from truthfulqa.presets import preset_map, COMPARE_PRIMER
from truthfulqa.models import find_subsequence, set_columns, MC_calcs
from truthfulqa.evaluates import format_frame, data_to_dict


def load_nq():
    dataset = load_dataset("OamPatel/iti_nq_open_val")["validation"]
    df = pd.DataFrame(columns=["question", "answer", "false_answer"])
    for row in dataset:
        new_row = pd.DataFrame({"question": [row["question"]], "answer": [[_ for _ in row["answer"]]], "false_answer": [row["false_answer"]]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df

def load_triviaqa():
    dataset = load_dataset("OamPatel/iti_trivia_qa_val")["validation"]
    df = pd.DataFrame(columns=["question", "answer", "false_answer"])
    for row in dataset:
        new_row = pd.DataFrame({"question": [row["question"]], "answer": [[_ for _ in row["answer"]['aliases']]], "false_answer": [row["false_answer"]]})
        df = pd.concat([df, new_row], ignore_index=True)
    return df

def format_truthfulqa(question, choice):
    return f"Q: {question} A: {choice}"

def format_truthfulqa_end_q(question, choice, rand_question): 
    return f"Q: {question} A: {choice} Q: {rand_question}"

def tokenized_tqa(dataset, tokenizer): 

    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        question = dataset[i]['question']
        choices = dataset[i]['mc2_targets']['choices']
        labels = dataset[i]['mc2_targets']['labels']

        assert len(choices) == len(labels), (len(choices), len(labels))

        for j in range(len(choices)): 
            choice = choices[j]
            label = labels[j]
            prompt = format_truthfulqa(question, choice)
            if i == 0 and j == 0: 
                print(prompt)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(label)
    
    return all_prompts, all_labels

def tokenized_tqa_gen_end_q(dataset, tokenizer): 

    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)): 
        question = dataset[i]['question']
        category = dataset[i]['category']
        rand_idx = np.random.randint(len(dataset))
        rand_question = dataset[rand_idx]['question']

        for j in range(len(dataset[i]['correct_answers'])): 
            answer = dataset[i]['correct_answers'][j]
            prompt = format_truthfulqa_end_q(question, answer, rand_question)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(1)
            all_categories.append(category)
        
        for j in range(len(dataset[i]['incorrect_answers'])):
            answer = dataset[i]['incorrect_answers'][j]
            prompt = format_truthfulqa_end_q(question, answer, rand_question)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(0)
            all_categories.append(category)
        
    return all_prompts, all_labels, all_categories

def tokenized_tqa_gen(dataset, tokenizer): 

    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)): 
        question = dataset[i]['question']
        category = dataset[i]['category']

        for j in range(len(dataset[i]['correct_answers'])): 
            answer = dataset[i]['correct_answers'][j]
            prompt = format_truthfulqa(question, answer)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(1)
            all_categories.append(category)
        
        for j in range(len(dataset[i]['incorrect_answers'])):
            answer = dataset[i]['incorrect_answers'][j]
            prompt = format_truthfulqa(question, answer)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(0)
            all_categories.append(category)
        
    return all_prompts, all_labels, all_categories

def tokenize_toxicity_dataset(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    all_scores = []
    all_categories = []

    label_map = {
        "hate": 1,
        "neutral": 0
    }

    for i in range(len(dataset)):
        text = dataset[i]['text']
        label = dataset[i]['label']
        score = dataset[i]['toxicity_score']
        category = dataset[i]['target_groups']

        # Convert label string to binary int
        binary_label = label_map.get(label.lower(), 0)

        # Tokenize the text
        prompt = tokenizer(text, return_tensors='pt').input_ids

        all_prompts.append(prompt)
        all_labels.append(binary_label)
        all_scores.append(score)
        all_categories.append(category)

    return all_prompts, all_labels, all_scores, all_categories

def tokenize_toxigen(dataset, dataset_non, tokenizer):
    all_prompts = []
    all_labels = []
    all_text = []
    print("start tokenizing")
    for i in tqdm(range(len(dataset))):
        toxic_text = dataset['toxic paraphrase'][i]['text']
        non_toxic_text = dataset['non toxic paraphrase'][i]['text']
        text = dataset['original'][i]
        toxic_prompt = tokenizer(toxic_text, return_tensors='pt').input_ids
        non_toxic_prompt = tokenizer(non_toxic_text, return_tensors='pt').input_ids
        if len(toxic_prompt[0]) == 1 or len(non_toxic_prompt[0]) == 1:
            continue
        toxic_label = 1
        non_toxic_label = 0
        all_prompts.append(toxic_prompt)
        all_labels.append(toxic_label)
        all_prompts.append(non_toxic_prompt)
        all_labels.append(non_toxic_label)
        all_text.append((text, toxic_text, non_toxic_text))
    
    if not dataset_non:
        return all_prompts, all_labels, all_text
        
    for i in tqdm(range(len(dataset_non))):
        toxic_text = dataset_non['toxic paraphrase'][i]['text']
        non_toxic_text = dataset_non['non toxic paraphrase'][i]['text']
        text = dataset_non['original'][i]
        toxic_prompt = tokenizer(toxic_text, return_tensors='pt').input_ids
        non_toxic_prompt = tokenizer(non_toxic_text, return_tensors='pt').input_ids
        if len(toxic_prompt[0]) == 1 or len(non_toxic_prompt[0]) == 1:
            continue
        toxic_label = 1
        non_toxic_label = 0
        all_prompts.append(toxic_prompt)
        all_labels.append(toxic_label)
        all_prompts.append(non_toxic_prompt)
        all_labels.append(non_toxic_label)
        all_text.append((text, toxic_text, non_toxic_text))

    return all_prompts, all_labels, all_text

def tokenize_paradetox(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    all_text = []
    print("start tokenizing")
    for i in tqdm(range(len(dataset))):
        toxic_text = dataset[i]['toxic_text']
        non_toxic_text = dataset[i]['non_toxic_text']
        text = dataset[i]['text']
        toxic_prompt = tokenizer(toxic_text, return_tensors='pt').input_ids
        non_toxic_prompt = tokenizer(non_toxic_text, return_tensors='pt').input_ids
        if len(toxic_prompt[0]) == 1 or len(non_toxic_prompt[0]) == 1:
            continue
        toxic_label = 1
        non_toxic_label = 0
        all_prompts.append(toxic_prompt)
        all_labels.append(toxic_label)
        all_prompts.append(non_toxic_prompt)
        all_labels.append(non_toxic_label)
        all_text.append((text, toxic_text, non_toxic_text))

    return all_prompts, all_labels, all_text

def get_llama_activations_bau(model, prompt, device): 
    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS+MLPS) as ret:
        # with TraceDict(model, HEADS+MLPS, retain_input=True) as ret:
            output = model(prompt, output_hidden_states = True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states

def get_gpt2_activations_pyvene(collected_model, collectors, prompt, tokenizer, device):
    """
    Runs a prompt through an IntervenableModel with Collector hooks and extracts:
    - layer-wise [num_layers x hidden_dim]
    - head-wise [num_layers x num_heads x head_dim]
    """

    # Reset hooks
    for collector in collectors:
        collector.reset()

    # Tokenize prompt and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Run model (triggers intervention hooks)
    _ = collected_model.model(**inputs)

    # Collect layer-wise activations
    layer_wise = [collector.states[0] for collector in collectors]  # each shape: [seq_len, hidden_dim]
    layer_wise = torch.stack(layer_wise, dim=0)                          # shape: [num_layers, seq_len, hidden_dim]

    # Only keep the last token’s activation
    layer_wise = layer_wise[:, -1, :]                                    # shape: [num_layers, hidden_dim]

    # Split hidden into heads (e.g., 12 heads, 768 -> [12, 64])
    num_heads = collected_model.model.config.num_attention_heads
    head_dim = layer_wise.shape[-1] // num_heads
    head_wise = layer_wise.view(layer_wise.shape[0], num_heads, head_dim)  # [num_layers, num_heads, head_dim]

    return layer_wise.numpy(), head_wise.numpy(), inputs


def get_gemma_activations_pyvene(collected_model, collectors, prompt, device):
    with torch.no_grad():
        prompt = prompt.to(device)
        output = collected_model({"input_ids": prompt, "output_hidden_states": True})[1]
    hidden_states = output.hidden_states
    hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
    hidden_states = hidden_states.detach().cpu().numpy()
    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            states_per_gen = torch.stack(collector.states, axis=0).cpu().numpy()
            head_wise_hidden_states.append(states_per_gen)
        else:
            head_wise_hidden_states.append(None)
        collector.reset()
    mlp_wise_hidden_states = []
    # print("SHAPES", len(head_wise_hidden_states), len(head_wise_hidden_states[0]))
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).squeeze().numpy()
    # print("SHAPES", head_wise_hidden_states.shape)
    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states


def get_llama_activations_pyvene(collected_model, collectors, prompt, device):
    with torch.no_grad():
        prompt = prompt.to(device)
        output = collected_model({"input_ids": prompt, "output_hidden_states": True})[1]
    hidden_states = output.hidden_states
    hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
    hidden_states = hidden_states.detach().cpu().numpy()
    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            states_per_gen = torch.stack(collector.states, axis=0).cpu().numpy()
            head_wise_hidden_states.append(states_per_gen)
        else:
            head_wise_hidden_states.append(None)
        collector.reset()
    mlp_wise_hidden_states = []
    # print("SHAPES", len(head_wise_hidden_states), len(head_wise_hidden_states[0]))
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).squeeze().numpy()
    # print("SHAPES", head_wise_hidden_states.shape)
    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states




