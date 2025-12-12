import copy, torch, random
import numpy as np
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer

from einops import rearrange
import pickle
import os
from tqdm import tqdm
import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from vae import VAE, vae_loss_function, train_vae, test_vae
from TruthfulQA.truthfulqa import utilities, models, metrics
from TruthfulQA.truthfulqa.configs import BEST_COL, ANSWER_COL, INCORRECT_COL
import gc


import sys
sys.path.append('../')
from utils_toxic import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions, get_activations
from utils_toxic import get_special_directions, get_matrix_directions, train_vae_and_extract_mu, get_top_heads_pns

HF_NAMES = {
    'llama_1B': 'meta-llama/Llama-3.2-1B', #meta-llama/Llama-3.2-1B
    'llama_3B': 'meta-llama/Llama-3.2-3B',
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf', 
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'vicuna_13B': 'lmsys/vicuna-13b-v1.5',
    'vicuna_pns': '/work/hdd/bcxt/yian3/models/vicuna_pns_finetuned',
    'COV_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_finetuned_epoch5',
    'COV_pns_use_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_True_finetuned_epoch5',
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'gemma3_4B': 'google/gemma-3-4b-it',
    'vicuna_13B_toxigen_vicuna_72_0.01_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_72_True_0.01_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_72_0.01_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_accuracy_72_False_0.01_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_18_0.01_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_True_0.01_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_36_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_36_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_accuracy_36_False_0.0001_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_18_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_True_0.0001_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_18_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_accuracy_18_False_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_72_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_accuracy_72_False_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_72_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_72_True_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_36_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_accuracy_36_False_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_36_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_18_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_accuracy_18_False_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_18_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_18_True_0.0001_finetuned_epoch5',
    'llama3_8B_hate_vicuna_18_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_hate_vicuna_accuracy_18_False_0.0001_finetuned_epoch5',
    'llama3_8B_hate_vicuna_18_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_hate_vicuna_logpns_18_True_0.0001_finetuned_epoch5',
    'llama3_8B_hate_vicuna_36_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_hate_vicuna_accuracy_36_False_0.0001_finetuned_epoch5',
    'llama3_8B_hate_vicuna_36_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_hate_vicuna_logpns_36_True_0.0001_finetuned_epoch5',
    'mistral_7B': 'mistralai/Mistral-7B-v0.1',
    'vicuna_13B_hate_vicuna_36_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_hate_vicuna_accuracy_36_False_0.0001_finetuned_epoch5',
    'vicuna_13B_hate_vicuna_36_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_hate_vicuna_logpns_36_True_0.0001_finetuned_epoch5',  
    'vicuna_13B_hate_vicuna_18_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_hate_vicuna_accuracy_18_False_0.0001_finetuned_epoch5',
    'vicuna_13B_hate_vicuna_18_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_hate_vicuna_logpns_18_True_0.0001_finetuned_epoch5',  
    'llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_bce_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_bce_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_l2_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_l2_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5',
}

def _center(X): return X - X.mean(0, keepdim=True)

def _unpack_batch(batch, device):
    if isinstance(batch, dict):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", None)
    else:
        if len(batch) == 3:  # Has labels
            ids, mask, labels = batch
            ids = ids.to(device); mask = mask.to(device); labels = labels.to(device)
        else:  # No labels
            ids, mask = batch
            ids = ids.to(device); mask = mask.to(device)
            labels = None
    return ids, mask, labels

def linear_cka(X, Y):
    Xc, Yc = _center(X), _center(Y)
    HSIC = (Xc.T @ Yc).pow(2).sum()
    K = (Xc.T @ Xc); L = (Yc.T @ Yc)
    denom = (K.pow(2).sum() * L.pow(2).sum()).sqrt() + 1e-12
    return (HSIC / denom).item()

def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_model_for_analysis(model):
    # Make outputs you’ll need available & deterministic
    model.eval()
    model.config.return_dict = True
    model.config.output_hidden_states = True
    model.config.output_attentions = True
    model.config.use_cache = False
    return model

def load_before_after(base_ckpt_or_dir, finetuned_ckpt_dir, device="cuda", dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(base_ckpt_or_dir, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Load models with memory optimization
    print("Loading base model...")
    model_before = AutoModelForCausalLM.from_pretrained(
        base_ckpt_or_dir, 
        torch_dtype=dtype,
        device_map="auto",  # Automatically handle device placement
        low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
    )
    
    print("Loading fine-tuned model...")
    model_after = AutoModelForCausalLM.from_pretrained(
        finetuned_ckpt_dir, 
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    for m in (model_before, model_after):
        prepare_model_for_analysis(m)

    return tok, model_before, model_after

def make_eval_loader(texts, tokenizer, max_len=512, batch_size=8):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
    return DataLoader(ds, batch_size=batch_size)

def make_toxicity_loader(dataset_path, tokenizer, max_len=512, batch_size=8, max_samples=None):
    """Create a loader with toxic and non-toxic texts with proper labels"""
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    if max_samples is not None:
        df = df.head(max_samples)
    
    # Create lists for toxic and non-toxic texts
    toxic_texts = []
    non_toxic_texts = []
    labels = []
    
    for _, row in df.iterrows():
        # Add instruction to both toxic and non-toxic texts
        toxic_text = f"""Please continue the following sentence: "{row['toxic_text']}" """
        non_toxic_text = f"""Please continue the following sentence: "{row['non_toxic_text']}" """
        
        toxic_texts.append(toxic_text)
        non_toxic_texts.append(non_toxic_text)
        labels.extend([1, 0])  # 1 for toxic, 0 for non-toxic
    
    # Combine all texts
    all_texts = []
    for i in range(len(toxic_texts)):
        all_texts.append(toxic_texts[i])
        all_texts.append(non_toxic_texts[i])
    
    # Tokenize
    enc = tokenizer(all_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    
    # Create dataset with labels
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels))
    return DataLoader(ds, batch_size=batch_size)

# Optional: hooks if you want attn o_proj inputs/outputs for head-level analysis
def attach_o_proj_hooks(model, store_dict, layers=None, capture="out"):  # capture: "in" or "out"
    """
    store_dict[L] -> (B,T,H) tensor from self_attn.o_proj (input or output)
    """
    handles = []
    total_layers = len(model.model.layers)
    target_layers = range(total_layers) if layers is None else layers

    def mk_hook(L):
        def hook(mod, inputs, output):
            x = output if capture == "out" else inputs[0]
            store_dict[L] = x  # Don't detach; this is analysis (no backward)
        return hook

    for L in target_layers:
        h = model.model.layers[L].self_attn.o_proj.register_forward_hook(mk_hook(L))
        handles.append(h)
    return handles

def load_texts_from_dataset(dataset_name, model_name, max_samples=None):
    """Load texts from the dataset files similar to validate_2fold_toxic.py"""
    seed = 2
    if dataset_name == "toxigen_vicuna" or dataset_name == "hate_vicuna":
        input_path = f'/u/yian3/unlearn_LLM_toxicity/causal_to_concept/llms/splits/{dataset_name}_fold_0_test_seed_{seed}.csv'   
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    questions = utilities.load_questions(filename=input_path)
    
    # Extract text column and format with instruction
    raw_texts = questions['text'].tolist()
    
    # Format each text with the instruction
    formatted_texts = []
    for text in raw_texts:
        formatted_text = f"""Please continue the following sentence: "{text}" """
        formatted_texts.append(formatted_text)
    
    if max_samples is not None:
        formatted_texts = formatted_texts[:max_samples]
    
    print(f"Loaded and formatted {len(formatted_texts)} texts")
    return formatted_texts

def extract_activations_batch(model, dataloader, device, selected_heads=None, layers=None):
    """Extract activations from model for representation analysis"""
    model.eval()
    all_hidden_states = []
    all_attention_states = []
    all_o_proj_inputs = {}
    all_o_proj_outputs = {}
    
    # Get model dtype
    model_dtype = next(model.parameters()).dtype
    
    # Attach hooks if selected_heads is provided
    o_proj_handles = []
    if selected_heads is not None:
        o_proj_handles = attach_o_proj_hooks(model, all_o_proj_inputs, capture="in")
        o_proj_handles.extend(attach_o_proj_hooks(model, all_o_proj_outputs, capture="out"))
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting activations"):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True
            )
            
            # Store hidden states (all layers)
            if layers is None:
                # Store all layers
                all_hidden_states.append([h.cpu() for h in outputs.hidden_states])
            else:
                # Store only specified layers
                selected_hidden = [outputs.hidden_states[i].cpu() for i in layers]
                all_hidden_states.append(selected_hidden)
            
            # Store attention states
            all_attention_states.append([att.cpu() for att in outputs.attentions])
    
    # Remove hooks
    for handle in o_proj_handles:
        handle.remove()
    
    return all_hidden_states, all_attention_states, all_o_proj_inputs, all_o_proj_outputs

def compare_representations(model_before, model_after, dataloader, device, selected_heads=None, layers=None):
    """Compare representations between before and after fine-tuning"""
    print("Extracting activations from model_before...")
    hidden_before, attn_before, o_proj_in_before, o_proj_out_before = extract_activations_batch(
        model_before, dataloader, device, selected_heads, layers
    )
    
    print("Extracting activations from model_after...")
    hidden_after, attn_after, o_proj_in_after, o_proj_out_after = extract_activations_batch(
        model_after, dataloader, device, selected_heads, layers
    )
    
    return {
        'hidden_before': hidden_before,
        'hidden_after': hidden_after,
        'attn_before': attn_before,
        'attn_after': attn_after,
        'o_proj_in_before': o_proj_in_before,
        'o_proj_in_after': o_proj_in_after,
        'o_proj_out_before': o_proj_out_before,
        'o_proj_out_after': o_proj_out_after
    }

def compute_representation_similarity(activations_before, activations_after, metric='cosine'):
    """Compute similarity between before/after representations"""
    similarities = {}
    
    for layer_idx in range(len(activations_before[0])):  # Assuming same number of layers
        layer_sims = []
        
        for batch_idx in range(len(activations_before)):
            before = activations_before[batch_idx][layer_idx]  # [B, T, D]
            after = activations_after[batch_idx][layer_idx]   # [B, T, D]
            
            # Flatten to [B*T, D] for comparison
            before_flat = before.view(-1, before.size(-1))
            after_flat = after.view(-1, after.size(-1))
            
            if metric == 'cosine':
                # Cosine similarity
                sim = torch.nn.functional.cosine_similarity(before_flat, after_flat, dim=1)
                layer_sims.append(sim.mean().item())
            elif metric == 'l2':
                # L2 distance
                dist = torch.norm(before_flat - after_flat, dim=1)
                layer_sims.append(dist.mean().item())
        
        similarities[f'layer_{layer_idx}'] = np.mean(layer_sims)
    
    return similarities

def analyze_head_changes(o_proj_before, o_proj_after, selected_heads, model_config):
    """Analyze changes in specific attention heads"""
    head_analysis = {}
    
    for layer_idx, head_idx in selected_heads:
        if layer_idx in o_proj_before and layer_idx in o_proj_after:
            before = o_proj_before[layer_idx]  # [B, T, H*D]
            after = o_proj_after[layer_idx]    # [B, T, H*D]
            
            # Extract specific head
            head_dim = model_config.hidden_size // model_config.num_attention_heads
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            
            before_head = before[:, :, start:end]  # [B, T, D]
            after_head = after[:, :, start:end]    # [B, T, D]
            
            # Compute changes
            cosine_sim = torch.nn.functional.cosine_similarity(
                before_head.view(-1, head_dim), 
                after_head.view(-1, head_dim), 
                dim=1
            ).mean().item()
            
            l2_dist = torch.norm(before_head - after_head, dim=-1).mean().item()
            
            head_analysis[f'layer_{layer_idx}_head_{head_idx}'] = {
                'cosine_similarity': cosine_sim,
                'l2_distance': l2_dist,
                'change_magnitude': l2_dist / (torch.norm(before_head, dim=-1).mean().item() + 1e-8)
            }
    
    return head_analysis

@torch.no_grad()
def per_token_nll_means(model, loader, device):
    prepare_model_for_analysis(model)
    T = None
    sum_nll, cnt = None, None
    total_nll, total_tok = 0.0, 0
    for batch in loader:
        ids, mask, _ = _unpack_batch(batch, device)
        out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits[:, :-1, :].float()
        target = ids[:, 1:]
        m = mask[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                              target.reshape(-1),
                              reduction="none").view_as(target)
        if T is None:
            T = target.size(1)
            sum_nll = torch.zeros(T, device=device)
            cnt = torch.zeros(T, device=device)
        sum_nll += (nll * m).sum(dim=0)
        cnt += m.sum(dim=0)
        total_nll += (nll * m).sum().item()
        total_tok += m.sum().item()
    mean_nll = (sum_nll / cnt.clamp_min(1)).cpu().numpy()
    ppl = math.exp(total_nll / max(total_tok, 1))
    return mean_nll, ppl

def plot_pos_nll(means_before, means_after, save_path):
    T = len(means_before)
    x = np.arange(T)
    delta = means_after - means_before
    plt.figure(figsize=(10,4))
    plt.plot(x, means_before, label="Before")
    plt.plot(x, means_after,  label="After")
    plt.title("Per-position NLL (lower is better)")
    plt.xlabel("Position"); plt.ylabel("NLL")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_path, "pos_nll_before_after.png")); plt.close()

    plt.figure(figsize=(10,3))
    plt.plot(x, delta)
    plt.axhline(0, ls="--", lw=1, color="gray")
    plt.title("NLL Δ (After − Before) by position")
    plt.xlabel("Position"); plt.ylabel("Δ NLL")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "pos_nll_delta.png")); plt.close()

@torch.no_grad()
def layerwise_logit_ce(model, loader, device, layers=None):
    prepare_model_for_analysis(model)
    L_total = len(model.model.layers)
    layers = layers or list(range(L_total))
    sums = {L: 0.0 for L in layers}
    counts = 0
    W_U = model.get_output_embeddings().weight.float()  # (V,D)
    for batch in loader:
        ids, mask, _ = _unpack_batch(batch, device)
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hs = out.hidden_states  # [emb, L0, L1, ...]
        target = ids[:, 1:]
        m = mask[:, 1:].float()
        for L in layers:
            H = hs[L+1][:, :-1, :].float()          # align to next token
            logits = H @ W_U.T
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   target.reshape(-1), reduction="none").view_as(m)
            sums[L] += (loss * m).sum().item()
        counts += m.sum().item()
    return {L: sums[L] / max(counts,1) for L in layers}

def plot_logit_lens(ce_b, ce_a, save_path):
    layers = sorted(ce_b.keys())
    b = np.array([ce_b[L] for L in layers])
    a = np.array([ce_a[L] for L in layers])
    
    # Plot 1: Both curves as line plots
    plt.figure(figsize=(10,4))
    plt.plot(layers, b, marker="o", label="Before", linewidth=2)
    plt.plot(layers, a, marker="s", label="After", linewidth=2)
    plt.xlabel("Layer"); plt.ylabel("Cross-Entropy")
    plt.title("Logit-lens CE by layer")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "logit_lens_ce.png")); plt.close()

    # Plot 2: Delta (difference)
    plt.figure(figsize=(10,3))
    plt.plot(layers, a - b, marker="o", linewidth=2)
    plt.axhline(0, ls="--", lw=1, color="gray")
    plt.xlabel("Layer"); plt.ylabel("Δ CE (After−Before)")
    plt.title("Logit-lens CE Δ")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "logit_lens_delta.png")); plt.close()

@torch.no_grad()
def collect_layer_tokens(model, loader, device, layers, stride=4, max_batches=8, max_tokens=20000):
    prepare_model_for_analysis(model)
    bufs = {L: [] for L in layers}
    tok_count = {L: 0 for L in layers}
    seen = 0
    for batch in loader:
        ids, mask, _ = _unpack_batch(batch, device)
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hs = out.hidden_states
        for L in layers:
            H = hs[L+1].float()                 # (B,T,D)
            M = mask.bool()
            H = H[:, ::stride, :]; M = M[:, ::stride]
            H = H[M].cpu()
            if H.numel() == 0: continue
            bufs[L].append(H)
            tok_count[L] += H.size(0)
            if tok_count[L] >= max_tokens: bufs[L] = [torch.cat(bufs[L], 0)[:max_tokens]]
        seen += 1
        if seen >= max_batches: break
    return {L: (torch.cat(bufs[L], 0) if len(bufs[L]) else torch.empty(0)) for L in layers}

def compute_cka_by_layer(model_b, model_a, loader, device, stride=4, max_batches=8):
    L_total = len(model_a.model.layers)
    layers = list(range(L_total))
    REP_b = collect_layer_tokens(model_b, loader, device, layers, stride, max_batches)
    REP_a = collect_layer_tokens(model_a, loader, device, layers, stride, max_batches)
    return {L: (linear_cka(REP_b[L], REP_a[L]) if REP_b[L].numel() else np.nan) for L in layers}

def plot_cka(cka_dict, save_path):
    layers = sorted(cka_dict.keys())
    vals = np.array([cka_dict[L] for L in layers])
    plt.figure(figsize=(10,3.5))
    plt.plot(layers, vals, marker="o")
    plt.ylim(0,1.05)
    plt.xlabel("Layer"); plt.ylabel("Linear CKA")
    plt.title("Representation similarity (Before vs After)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "cka_per_layer.png")); plt.close()

def oproj_head_delta_matrix(model_a, model_b):
    H  = model_a.config.hidden_size
    nH = model_a.config.num_attention_heads
    dH = H // nH
    L = len(model_a.model.layers)
    M = np.zeros((L, nH), dtype=np.float32)
    for Lidx in range(L):
        Wa = model_a.model.layers[Lidx].self_attn.o_proj.weight.data.float()
        Wb = model_b.model.layers[Lidx].self_attn.o_proj.weight.data.float()
        for h in range(nH):
            s,e = h*dH, (h+1)*dH
            M[Lidx, h] = torch.norm(Wa[:,s:e] - Wb[:,s:e], p='fro').item()
    return M

def plot_head_delta_heatmap(delta_matrix, save_path, title="o_proj head-slice Δ (Frobenius norm)"):
    plt.figure(figsize=(12, 6))
    plt.imshow(delta_matrix, aspect='auto')
    plt.colorbar(label="||ΔW||_F")
    plt.xlabel("Head"); plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "head_deltas_heatmap.png")); plt.close()

@torch.no_grad()
def attention_entropy_per_head(model, loader, device):
    prepare_model_for_analysis(model)
    L = len(model.model.layers)
    nH = model.config.num_attention_heads
    # accumulate batch-averaged head entropies
    sums = torch.zeros(L, nH, dtype=torch.float32)
    counts = torch.zeros(L, nH, dtype=torch.float32)

    for batch in loader:
        ids, mask, _ = _unpack_batch(batch, device)        # mask: (B, T)
        out = model(input_ids=ids, attention_mask=mask, output_attentions=True)

        for Lidx, att in enumerate(out.attentions):     # att: (B, H, T, S)
            p = att.float().clamp_min(1e-12)
            # entropy over keys S, normalized to [0,1]; result: (B, H, T)
            ent = -(p * p.log()).sum(dim=-1) / math.log(p.size(-1))

            # build mask for query positions only -> (B,1,T) then broadcast to (B,H,T)
            m = mask[:, :ent.size(-1)].unsqueeze(1).float()        # (B,1,T)
            ent_masked = ent * m                                   # (B,H,T)

            # average over batch & tokens for each head → (H,)
            head_mean = ent_masked.sum(dim=(0, 2)) / m.sum(dim=(0, 2)).clamp_min(1e-8)
            sums[Lidx] += head_mean.cpu()
            counts[Lidx] += 1

    return (sums / counts.clamp_min(1)).numpy()  # shape: (L, H)

def plot_entropy_delta(EntB, EntA, save_path):
    D = EntA - EntB  # (L,H)
    plt.figure(figsize=(12,6))
    plt.imshow(D, aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label="Δ Entropy (After−Before)")
    plt.xlabel("Head"); plt.ylabel("Layer"); plt.title("Attention entropy Δ")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "attention_entropy_delta.png")); plt.close()

@torch.no_grad()
def layer_geometry_stats(model, loader, device, stride=8, max_batches=6):
    prepare_model_for_analysis(model)
    L = len(model.model.layers)
    stats = {Lidx: {} for Lidx in range(L)}
    # Collect small token sets per layer
    REP = collect_layer_tokens(model, loader, device, layers=list(range(L)),
                               stride=stride, max_batches=max_batches, max_tokens=20000)
    for Lidx in range(L):
        H = REP[Lidx]
        if H.numel() == 0:
            stats[Lidx] = {"mean_norm": np.nan, "feat_var": np.nan, "anisotropy": np.nan}
            continue
        H = H.float()
        mean_norm = H.norm(dim=1).mean().item()
        Hc = H - H.mean(0, keepdim=True)
        feat_var = Hc.var(dim=0, unbiased=False).mean().item()
        # top-PC fraction via randomized power iteration
        v = torch.randn(H.size(1))
        for _ in range(6):
            v = (Hc.T @ (Hc @ v)); v = v / (v.norm()+1e-12)
        topvar = (Hc @ v).pow(2).mean().item()
        anis = topvar / (Hc.pow(2).sum(dim=1).mean().item() + 1e-12)
        stats[Lidx] = {"mean_norm": mean_norm, "feat_var": feat_var, "anisotropy": anis}
    return stats

def plot_geometry(stats_b, stats_a, save_path):
    layers = sorted(stats_b.keys())
    def arr(key, s): return np.array([s[L][key] for L in layers])
    for key, title, fname in [
        ("mean_norm", "Mean token L2 norm", "geom_mean_norm.png"),
        ("feat_var",  "Per-dim feature variance", "geom_feat_var.png"),
        ("anisotropy","Anisotropy (top-PC fraction)", "geom_anisotropy.png"),
    ]:
        b = arr(key, stats_b); a = arr(key, stats_a)
        plt.figure(figsize=(10,3.5))
        plt.plot(layers, b, label="Before"); plt.plot(layers, a, label="After")
        plt.xlabel("Layer"); plt.ylabel(key); plt.title(title)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_path, fname)); plt.close()

@torch.no_grad()
def collect_representations_with_labels(model, loader, device, layer_idx, max_samples=1000):
    """Collect representations from a specific layer with toxicity labels"""
    prepare_model_for_analysis(model)
    representations = []
    labels = []
    
    count = 0
    for batch in loader:
        if count >= max_samples:
            break
            
        ids, mask, batch_labels = _unpack_batch(batch, device)
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        
        # Get representations from specified layer
        hidden_states = out.hidden_states[layer_idx + 1]  # +1 because first is embedding
        
        # Extract representations (mean pooling over sequence length)
        for i in range(hidden_states.size(0)):
            if count >= max_samples:
                break
            # Mean pool over valid tokens
            valid_tokens = mask[i].bool()
            if valid_tokens.sum() > 0:
                pooled_repr = hidden_states[i][valid_tokens].mean(dim=0).cpu().numpy()
                representations.append(pooled_repr)
                
                # Use the proper label from the dataset
                if batch_labels is not None:
                    labels.append(batch_labels[i].item())
                else:
                    # Fallback to text-based detection if no labels
                    text = model.config.tokenizer.decode(ids[i][valid_tokens], skip_special_tokens=True)
                    if "toxic" in text.lower():
                        labels.append(1)  # Toxic
                    else:
                        labels.append(0)  # Non-toxic
                count += 1
    
    return np.array(representations), np.array(labels)

def plot_representation_space(model_before, model_after, loader, device, save_path, 
                            layer_idx=16, max_samples=1000, method='tsne'):
    """Visualize representation space for toxic vs non-toxic content"""
    print(f"Collecting representations from layer {layer_idx}...")
    
    # Collect representations from both models
    repr_before, labels_before = collect_representations_with_labels(
        model_before, loader, device, layer_idx, max_samples
    )
    repr_after, labels_after = collect_representations_with_labels(
        model_after, loader, device, layer_idx, max_samples
    )
    
    # Ensure we have the same labels for both models
    min_len = min(len(labels_before), len(labels_after))
    labels = labels_before[:min_len]
    repr_before = repr_before[:min_len]
    repr_after = repr_after[:min_len]
    
    print(f"Collected {len(repr_before)} representations")
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(repr_before)//4))
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
    
    print("Computing dimensionality reduction...")
    repr_before_2d = reducer.fit_transform(repr_before)
    repr_after_2d = reducer.fit_transform(repr_after)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot before fine-tuning
    toxic_mask = labels == 1
    non_toxic_mask = labels == 0
    
    axes[0].scatter(repr_before_2d[non_toxic_mask, 0], repr_before_2d[non_toxic_mask, 1], 
                   c='blue', alpha=0.6, s=20, label='Non-toxic')
    axes[0].scatter(repr_before_2d[toxic_mask, 0], repr_before_2d[toxic_mask, 1], 
                   c='red', alpha=0.6, s=20, label='Toxic')
    axes[0].set_title(f'Before Fine-tuning (Layer {layer_idx})')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot after fine-tuning
    axes[1].scatter(repr_after_2d[non_toxic_mask, 0], repr_after_2d[non_toxic_mask, 1], 
                   c='blue', alpha=0.6, s=20, label='Non-toxic')
    axes[1].scatter(repr_after_2d[toxic_mask, 0], repr_after_2d[toxic_mask, 1], 
                   c='red', alpha=0.6, s=20, label='Toxic')
    axes[1].set_title(f'After Fine-tuning (Layer {layer_idx})')
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"representation_space_{method}_layer_{layer_idx}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a combined plot showing the shift
    plt.figure(figsize=(10, 8))
    
    # Plot before (lighter colors)
    plt.scatter(repr_before_2d[non_toxic_mask, 0], repr_before_2d[non_toxic_mask, 1], 
               c='lightblue', alpha=0.4, s=15, label='Non-toxic (Before)')
    plt.scatter(repr_before_2d[toxic_mask, 0], repr_before_2d[toxic_mask, 1], 
               c='lightcoral', alpha=0.4, s=15, label='Toxic (Before)')
    
    # Plot after (darker colors)
    plt.scatter(repr_after_2d[non_toxic_mask, 0], repr_after_2d[non_toxic_mask, 1], 
               c='blue', alpha=0.7, s=20, label='Non-toxic (After)')
    plt.scatter(repr_after_2d[toxic_mask, 0], repr_after_2d[toxic_mask, 1], 
               c='red', alpha=0.7, s=20, label='Toxic (After)')
    
    plt.title(f'Representation Space Shift (Layer {layer_idx}) - {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"representation_shift_{method}_layer_{layer_idx}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Representation space plots saved for layer {layer_idx}")

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Fine-tuning representation analysis')
    parser.add_argument('--base_model', type=str, required=True, help='Base model name or path')
    parser.add_argument('--finetuned_model', type=str, required=True, help='Fine-tuned model path')
    parser.add_argument('--dataset_name', type=str, default='toxigen_vicuna', help='Dataset name')
    parser.add_argument('--max_samples', type=int, default=50, help='Maximum samples to analyze')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for analysis')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--heads_path', type=str, help='Path to selected heads file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load models
    print(f"Loading base model: {args.base_model}")
    print(f"Loading fine-tuned model: {args.finetuned_model}")
    
    tokenizer, model_before, model_after = load_before_after(
        HF_NAMES[args.base_model], 
        HF_NAMES[args.finetuned_model], 
        device=args.device, 
        dtype=torch.float16
    )
    
    # Clear cache and garbage collect
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load texts
    print(f"Loading texts from {args.dataset_name}...")
    texts = load_texts_from_dataset(args.dataset_name, args.base_model, args.max_samples)
    
    # Create dataloader
    eval_loader = make_eval_loader(texts, tokenizer, max_len=args.max_length, batch_size=args.batch_size)
    
    # Load selected heads if provided
    selected_heads = None
    if args.heads_path and os.path.exists(args.heads_path):
        selected_heads = np.load(args.heads_path)
        print(f"Loaded {len(selected_heads)} selected heads")
    
    # Extract and compare representations
    print("Starting representation analysis...")
    activations = compare_representations(
        model_before, model_after, eval_loader, args.device
    )
    # Compute similarities
    print("Computing representation similarities...")
    hidden_similarities = compute_representation_similarity(
        activations['hidden_before'], activations['hidden_after'], metric='cosine'
    )
    
    # Print selected heads information
    if selected_heads is not None:
        # Get unique layers from selected heads
        unique_layers = sorted(list(set([layer for layer, head in selected_heads])))
        print(f"\nUnique layers in selected heads: {unique_layers}")
        print(f"Number of unique layers: {len(unique_layers)}")
    
    # Analyze head changes if selected heads provided
    head_analysis = {}
    if selected_heads is not None:
        print("\nAnalyzing head-specific changes...")
        head_analysis = analyze_head_changes(
            activations['o_proj_out_before'], 
            activations['o_proj_out_after'], 
            selected_heads, 
            model_before.config
        )
    
    # Print results
    # print("\n=== Representation Analysis Results ===")
    # print("\nHidden State Similarities (Cosine) - All Layers:")
    # for layer, sim in hidden_similarities.items():
    #     print(f"  {layer}: {sim:.4f}")
    
    # Print similarities for selected head layers specifically
    if selected_heads is not None:
        print(f"\nHidden State Similarities for Selected Head Layers:")
        for layer, head in selected_heads:
            layer_key = f'layer_{layer}'
            if layer_key in hidden_similarities:
                print(f"  Layer {layer} (used by head {head}): {hidden_similarities[layer_key]:.4f}")
    
    if head_analysis:
        print("\nHead-Specific Changes:")
        for head, metrics in head_analysis.items():
            print(f"  {head}:")
            print(f"    Cosine Similarity: {metrics['cosine_similarity']:.4f}")
            print(f"    L2 Distance: {metrics['l2_distance']:.4f}")
            print(f"    Change Magnitude: {metrics['change_magnitude']:.4f}")
    
    # Create model-specific save directory
    base_model_name = args.base_model.replace('/', '_').replace('-', '_')
    finetuned_model_name = args.finetuned_model.replace('/', '_').replace('-', '_')
    plots_dir = getattr(args, "plots_dir", f"./analysis_plots_{base_model_name}_{finetuned_model_name}")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Analysis plots will be saved to: {os.path.abspath(plots_dir)}")

    # # 1) Per-position NLL and deltas
    # print("\n[1/6] Per-position NLL ...")
    # means_b, ppl_b = per_token_nll_means(model_before, eval_loader, args.device)
    # means_a, ppl_a = per_token_nll_means(model_after,  eval_loader, args.device)
    # print(f"Perplexity Before: {ppl_b:.3f} | After: {ppl_a:.3f} | Δ: {ppl_a - ppl_b:+.3f}")
    # plot_pos_nll(means_b, means_a, plots_dir)

    # # 2) Logit-lens CE per layer
    # print("[2/6] Logit-lens CE per layer ...")
    # ce_b = layerwise_logit_ce(model_before, eval_loader, args.device)
    # ce_a = layerwise_logit_ce(model_after,  eval_loader, args.device)
    # plot_logit_lens(ce_b, ce_a, plots_dir)

    # # 3) Linear CKA per layer
    # print("[3/6] Linear CKA per layer ...")
    # cka = compute_cka_by_layer(model_before, model_after, eval_loader, args.device, stride=4, max_batches=8)
    # plot_cka(cka, plots_dir)

    # # 4) o_proj head-slice weight deltas
    # print("[4/6] Head-slice weight deltas ...")
    # delta_mat = oproj_head_delta_matrix(model_after, model_before)  # After vs Before
    # plot_head_delta_heatmap(delta_mat, plots_dir)

    # # 5) Attention entropy Δ (optional; can be slow)
    # print("[5/6] Attention entropy Δ ...")
    # EntB = attention_entropy_per_head(model_before, eval_loader, args.device)
    # EntA = attention_entropy_per_head(model_after,  eval_loader, args.device)
    # plot_entropy_delta(EntB, EntA, plots_dir)

    # # 6) Geometry drift
    # print("[6/7] Geometry drift ...")
    # geom_b = layer_geometry_stats(model_before, eval_loader, args.device)
    # geom_a = layer_geometry_stats(model_after,  eval_loader, args.device)
    # plot_geometry(geom_b, geom_a, plots_dir)

    # 7) Representation space visualization
    print("[7/7] Representation space visualization ...")
    
    # Create toxicity-specific loader for representation analysis
    # Map model name to file naming pattern
    model_name_mapping = {
        'llama_3B': 'llama3_8B',  # Map the base model name to the file naming pattern
        'llama3_8B': 'llama3_8B',
        'vicuna_13B': 'vicuna_13B',
        'mistral_7B': 'mistral_7B'
    }
    file_model_name = model_name_mapping.get(args.base_model, args.base_model)
    
    toxicity_loader = make_toxicity_loader(
        f'/u/yian3/unlearn_LLM_toxicity/causal_to_concept/llms/splits/{args.dataset_name}_fold_0_test_seed_2.csv',
        tokenizer, 
        max_len=args.max_length, 
        batch_size=args.batch_size, 
        max_samples=100  # Use fewer samples for visualization
    )
    
    # Visualize for a few key layers (early, middle, late)
    key_layers = [8, 16, 24]  # Adjust based on your model's total layers
    for layer_idx in key_layers:
        if layer_idx < len(model_before.model.layers):
            plot_representation_space(model_before, model_after, toxicity_loader, args.device, 
                                    plots_dir, layer_idx=layer_idx, max_samples=200, method='tsne')
            plot_representation_space(model_before, model_after, toxicity_loader, args.device, 
                                    plots_dir, layer_idx=layer_idx, max_samples=200, method='pca')

    print(f"\nAll plots saved to: {os.path.abspath(plots_dir)}")

if __name__ == "__main__":
    main()

"""
python finetune_analysis.py --base_model llama3_8B \
--finetuned_model llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5 \
--dataset_name toxigen_vicuna --max_samples 100 --batch_size 8 --max_length 256 \
--heads_path /work/hdd/bcxt/yian3/toxic/features/heads/True_llama3_8B_toxigen_vicuna_seed_2_top_36_heads_alpha_15.0_fold_0_top_heads.npy --device cuda --seed 42

"""

# --heads_path /work/hdd/bcxt/yian3/toxic/features/heads/True_llama3_8B_toxigen_vicuna_seed_2_top_72_heads_alpha_15.0_fold_0_top_heads.npy --device cuda --seed 42
