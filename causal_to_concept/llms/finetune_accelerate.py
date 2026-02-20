"""
Multi-GPU/Multi-Node Fine-tuning with Hugging Face Accelerate

This script distributes training across multiple GPUs/nodes using Accelerate,
significantly speeding up the fine-tuning process.

IMPORTANT FOR LARGE MODELS (like vicuna_13B):
    To avoid OOM errors, configure Accelerate to use FSDP (Fully Sharded Data Parallel):
    
    1. Run: accelerate config
    2. When asked about distributed training, select:
       - "Fully Sharded Data Parallel (FSDP)" 
       - Or "DeepSpeed ZeRO Stage 2/3"
    3. FSDP shards the model across GPUs instead of replicating it (like DDP)
    4. This reduces memory usage from ~26GB per GPU to ~6.5GB per GPU (for 4 GPUs)

Setup:
    1. Configure accelerate: accelerate config (choose FSDP for large models!)
    2. Choose multi-GPU or multi-node setup

Usage:
    # Single node, multiple GPUs (e.g., 4 GPUs)
    accelerate launch finetune_accelerate.py \
        --model_name vicuna_13B \
        --dataset_name toxigen_vicuna \
        --head_select logpns \
        --num_heads 36 \
        --use_pns \
        --epochs 5 \
        --use_kl --lambda_fm 0.001 --lambda_term2 1e-3 \
        --lr 1e-5 --use_l2 \
        --virtual_batch_size 256 --batch_size 1

    # Multi-node (2 nodes, 8 GPUs total)
    accelerate launch --multi_gpu --num_machines=2 --num_processes=8 \
        finetune_accelerate.py --model_name vicuna_13B ...

Speed Improvement:
    - 4 GPUs: ~4x faster than single GPU
    - 8 GPUs: ~8x faster than single GPU
    - Effective batch size = batch_size * num_gpus
"""

import copy
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np
from torch.cuda.amp import autocast
from math import ceil
from torch.utils.data import Subset
from torch.cuda.amp import GradScaler
from vae import VAE, vae_loss_function, train_vae, test_vae
from torch.utils.data import TensorDataset
import gc 
from utils_toxic import get_all_mu, evaluate_latent_mu
import warnings
import os
from accelerate import Accelerator
from accelerate.utils import set_seed
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
warnings.filterwarnings("ignore", message="`do_sample` is set to `False`")
warnings.filterwarnings("ignore", message="To copy construct from a tensor")
import argparse


# --- CONFIG ---
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
    'tiny_gpt2':"sshleifer/tiny-gpt2",
    'mistral-7b': "mistralai/Mistral-7B-v0.3",
    'mistral-7b-instruct': "mistralai/Mistral-7B-Instruct-v0.3",
    'gemma-2-9b': "google/gemma-2-9b",
    'gemma-2-9b-instruct':"google/gemma-2-9b-it",
    'vicuna_13B': 'lmsys/vicuna-13b-v1.5',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune language model with log-PNS objective (Multi-GPU)')
    
    # Model and dataset configuration
    parser.add_argument('--model_name', type=str, default="vicuna_13B",
                      help='String identifier for the model')
    parser.add_argument('--dataset_name', type=str, default="toxigen_vicuna",
                      help='Name of the dataset')
    parser.add_argument('--toxic_path', type=str, default="./dataset/vicuna-13b_toxic.json",
                      help='Path to toxic dataset')
    parser.add_argument('--nontoxic_path', type=str, default="./dataset/vicuna-13b_nontoxic.json",
                      help='Path to non-toxic dataset')
    parser.add_argument('--seed', type=int, default=2,
                      help='Seed for random number generator')
    # Training hyperparameters
    parser.add_argument('--lambda_reg', type=float, default=1e-2,
                      help='Regularization strength')
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size per GPU (will be multiplied by num_gpus)')
    parser.add_argument('--lr', type=float, default=1e-5,
                      help='Learning rate')
    parser.add_argument('--lambda_term2', type=float, default=1e-3,
                      help='Regularization strength for term 2')
    parser.add_argument('--lambda_cls', type=float, default=0.1,
                      help='Weight for classification loss term')
    parser.add_argument('--max_length', type=int, default=10,
                      help='Maximum sequence length')
    parser.add_argument('--virtual_batch_size', type=int, default=128,
                      help='Virtual batch size for gradient accumulation')
    
    # Model configuration
    parser.add_argument('--head_select', type=str, default='logpns',
                      help='Head selection method')
    parser.add_argument('--num_heads', type=int, default=18,
                      help='Number of heads to select')
    parser.add_argument('--heads_path', type=str, 
                      default="/work/hdd/bcxt/yian3/toxic/features/heads/True_vicuna_13B_toxigen_vicuna_seed_2_top_36_heads_alpha_5.0_fold_0.npy",
                      help='Path to selected heads numpy file')
    parser.add_argument('--alpha', type=float, default=5.0,
                      help='Alpha for logPNS')
    parser.add_argument('--use_pns', action='store_true', default=False)
    parser.add_argument('--use_l2', action='store_true', default=False,
                      help='Use L2 regularization on model weights')
    parser.add_argument('--l2_lambda', type=float, default=1e-4,
                      help='L2 regularization strength')
    parser.add_argument('--save_dir', type=str, 
                      default="/work/hdd/bcxt/yian3/toxic/models",
                      help='Directory to save model checkpoints')
    parser.add_argument('--use_kl', action='store_true', default=False,
                      help='Use KL divergence loss with teacher model')
    parser.add_argument('--lambda_fm', type=float, default=0.05,
                      help='Lambda for feature matching loss (default: 0.05)')
    return parser.parse_args()


# Import necessary functions from finetune_test.py
# (We'll copy the key functions here to avoid circular imports)
def _pick_fm_layers(model, selected_heads, args):
    if hasattr(args, "fm_layers") and args.fm_layers:
        return list(args.fm_layers)
    if selected_heads is not None and len(selected_heads) > 0:
        return sorted({L for (L, _) in selected_heads})
    L = len(model.model.layers)
    return [L//3, 2*L//3, L-1]


def _layernorm_lastdim(x):
    return F.layer_norm(x, x.shape[-1:])


def calculate_l2_loss(model, selected_heads, l2_lambda):
    """Calculate L2 regularization loss for selected model parameters."""
    l2_loss = 0.0
    for (layer_idx, head_idx) in selected_heads:
        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        l2_loss += torch.sum(o_proj.weight ** 2)
    return l2_lambda * l2_loss


def head_only_feature_loss(
    student_model,
    teacher_model,
    fwd_inp_s,
    fwd_inp_t,
    selected_pairs,
    attention_mask=None,
    take_last=True,
    normalize="layernorm",
    metric="l2"
):
    losses = []
    H = student_model.config.hidden_size
    nH = student_model.config.num_attention_heads
    dH = H // nH

    for (L, h) in selected_pairs:
        Xs = fwd_inp_s[L]
        Xt = fwd_inp_t[L]

        if Xs.dim() == 3 and take_last:
            Xs = Xs[:, -1, :]
            Xt = Xt[:, -1, :]

        start, end = h * dH, (h + 1) * dH
        hs = Xs[..., start:end].to(torch.float32)
        ht = Xt[..., start:end].to(torch.float32)

        Wo_s = student_model.model.layers[L].self_attn.o_proj.weight[:, start:end].to(torch.float32)
        Wo_t = teacher_model.model.layers[L].self_attn.o_proj.weight[:, start:end].to(torch.float32)

        ys = hs @ Wo_s.T
        yt = ht @ Wo_t.T

        if normalize == "layernorm":
            ys = F.layer_norm(ys, ys.shape[-1:])
            yt = F.layer_norm(yt, yt.shape[-1:])

        if metric == "cos":
            per = 1.0 - F.cosine_similarity(ys, yt, dim=-1, eps=1e-8)
        else:
            per = (ys - yt).pow(2).mean(dim=-1)

        if attention_mask is not None and Xs.dim() == 3 and not take_last:
            m = attention_mask.float()
            per = (per * m).sum(1) / (m.sum(1) + 1e-8)

        losses.append(per.mean())

    return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=hs.device)


class ToxicDataset_paired(Dataset):
    def __init__(self, json_path, tokenizer, max_len=10):
        # Use HuggingFace datasets library to handle JSONL automatically
        # load_dataset returns a DatasetDict with splits (train/test/val)
        # Even for a single file, it wraps it in {"train": Dataset(...)}
        # So we need ["train"] to access the actual Dataset
        toxic_data = load_dataset("json", data_files=json_path, split="train")
        
        # Extract texts and labels
        self.texts = []
        self.labels = []
        
        for item in toxic_data:
            # Handle paired format: toxic_text and non_toxic_text
            if 'toxic_text' in item and 'non_toxic_text' in item:
                # Skip blank entries
                if item["toxic_text"] != "(Blank)" and item["toxic_text"].strip():
                    # Add toxic text with label 1
                    self.texts.append(item["toxic_text"].strip('"'))  # Remove quotes if present
                    self.labels.append(1)  # 1 for toxic
                    
                    # Add non-toxic text with label 0
                    if item["non_toxic_text"] != "(Blank)" and item["non_toxic_text"].strip():
                        self.texts.append(item["non_toxic_text"].strip('"'))  # Remove quotes if present
                        self.labels.append(0)  # 0 for non-toxic
            elif 'text' in item:
                # If only 'text' field exists, use it with label from 'label' field or default to 0
                text = item['text']
                label = item.get('label', 0)
                if text and text.strip():
                    self.texts.append(text.strip())
                    self.labels.append(label)
            else:
                # Try to find text field
                text = next((v for k, v in item.items() if isinstance(v, str) and len(v) > 10), "")
                label = item.get('label', item.get('toxic', 0))
                if text and text.strip():
                    self.texts.append(text.strip())
                    self.labels.append(label)
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if len(self.texts) == 0:
            raise ValueError(f"No valid texts found in {json_path}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'index': torch.tensor(idx, dtype=torch.long)
        }


def setup_dataloaders(dataset, args):
    """Setup train and eval dataloaders."""
    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(0.8 * len(indices))
    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(Subset(dataset, eval_indices), batch_size=8, shuffle=False)
    return train_loader, eval_loader


def setup_model_parameters(model, selected_heads, args):
    """Setup which parameters to update and forward hooks."""
    params_to_update = []
    params_set = set()  # Track unique parameters to avoid duplicates
    forward_saved_inputs = defaultdict(lambda: None)
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            forward_saved_inputs[layer_idx] = input[0].detach()
        return hook_fn

    # Get unique layers first to avoid duplicate hooks
    unique_layers = set(layer_idx for (layer_idx, _) in selected_heads)
    
    # Register hooks for unique layers only
    for layer_idx in unique_layers:
        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        o_proj.register_forward_hook(make_hook(layer_idx))
    
    # Add parameters only once per layer (even if multiple heads from same layer)
    for layer_idx in unique_layers:
        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        # Use id() to track unique parameter objects
        if id(o_proj.weight) not in params_set:
            params_to_update.append(o_proj.weight)
            params_set.add(id(o_proj.weight))
        if o_proj.bias is not None and id(o_proj.bias) not in params_set:
            params_to_update.append(o_proj.bias)
            params_set.add(id(o_proj.bias))
    
    return params_to_update, forward_saved_inputs


def attach_o_proj_input_hooks(model, saved_inputs_dict, layers=None, with_grad=False):
    handles = []
    target_layers = layers if layers is not None else list(range(len(model.model.layers)))
    
    def make_hook(L):
        def hook(module, inp, out):
            saved_inputs_dict[L] = inp[0] if with_grad else inp[0].detach()
        return hook

    for L in target_layers:
        m = model.model.layers[L].self_attn.o_proj
        h = m.register_forward_hook(make_hook(L))
        handles.append(h)
    return handles


def train_model(accelerator, model, train_loader, eval_loader, optimizer, selected_heads, 
                params_to_update, forward_saved_inputs, args, tokenizer, teacher_model=None):
    """Main training loop with Accelerate."""
    if accelerator.is_local_main_process:
        print("[DEBUG] train_model() called - entering training loop")
    
    step = 0
    accumulated_samples = 0
    progress_bar = tqdm(
        total=args.epochs * ceil(len(train_loader) / (args.virtual_batch_size or 1)), 
        desc="Steps",
        disable=not accelerator.is_local_main_process
    )

    X_buffer = []
    Y_buffer = []
    C_buffer = []
    logits_buffer = []
    
    if accelerator.is_local_main_process:
        print("[DEBUG] Training set size:", len(train_loader.dataset))
        print("[DEBUG] Training batches:", len(train_loader))
    
    if args.use_kl and teacher_model is not None:
        if accelerator.is_local_main_process:
            print("[DEBUG] Setting up teacher model hooks...")
        teacher_forward_saved_inputs = defaultdict(lambda: None)
        teacher_handles = attach_o_proj_input_hooks(
            teacher_model, teacher_forward_saved_inputs, layers=None, with_grad=False
        )
        fm_layers = _pick_fm_layers(model, selected_heads, args)
        lambda_fm = args.lambda_fm
        fm_metric = getattr(args, "fm_metric", "cos")
        fm_stride = getattr(args, "fm_stride", 1)
        if accelerator.is_local_main_process:
            print("[DEBUG] Teacher model hooks set up")
    
    if accelerator.is_local_main_process:
        print("[DEBUG] Starting epoch loop...")
    for epoch in range(args.epochs):
        if accelerator.is_local_main_process:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        model.train()
        
        # Use tqdm for batch-level progress (only on main process)
        if accelerator.is_local_main_process:
            batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", 
                                 total=len(train_loader), leave=True)
        else:
            batch_progress = train_loader
            
        for batch in batch_progress:
            forward_saved_inputs.clear()
            input_ids = batch['input_ids'].to(accelerator.device)
            input_embeds = model.get_input_embeddings()(input_ids)
            attention_mask = batch['attention_mask'].to(accelerator.device)
            labels = batch['label'].to(accelerator.device).unsqueeze(1)
            indices = batch["index"].to(accelerator.device)
            
            with accelerator.autocast():
                outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True)
                logits = outputs.logits.squeeze(-1)
            
            # Teacher forward (only on main process to save memory)
            if args.use_kl and teacher_model is not None:
                with torch.no_grad():
                    if accelerator.is_main_process:
                        # Teacher is on CPU, move inputs to CPU
                        input_ids_cpu = input_ids.cpu()
                        attention_mask_cpu = attention_mask.cpu()
                        t_out = teacher_model(input_ids=input_ids_cpu, attention_mask=attention_mask_cpu, output_hidden_states=True)
                        
                        # Move teacher activations to GPU for feature matching
                        teacher_fwd_inp_gpu = {}
                        for L, _ in selected_heads:
                            if L in teacher_forward_saved_inputs and teacher_forward_saved_inputs[L] is not None:
                                teacher_fwd_inp_gpu[L] = teacher_forward_saved_inputs[L].to(accelerator.device)
                            else:
                                teacher_fwd_inp_gpu[L] = None
                    else:
                        teacher_fwd_inp_gpu = {L: None for L, _ in selected_heads}

                # Feature matching loss (only compute on main process, then broadcast)
                if accelerator.is_main_process:
                    fm_loss_batch = head_only_feature_loss(
                        student_model=model.module if hasattr(model, 'module') else model,
                        teacher_model=teacher_model,
                        fwd_inp_s=forward_saved_inputs,
                        fwd_inp_t=teacher_fwd_inp_gpu,
                        selected_pairs=selected_heads,
                        attention_mask=attention_mask,
                        take_last=True,
                        normalize="layernorm",
                        metric="l2",
                    )
                else:
                    fm_loss_batch = torch.tensor(0.0, device=accelerator.device)
                
                # Broadcast loss to all processes
                fm_loss_batch = accelerator.broadcast(fm_loss_batch, from_process=0)
                
                # Gradient accumulation for feature loss
                bs = input_ids.size(0)
                if args.virtual_batch_size:
                    scale = float(bs) / float(args.virtual_batch_size)
                else:
                    scale = 1.0
                
                # Use accelerator.backward() for distributed training
                scaled_fm_loss = lambda_fm * fm_loss_batch * scale
                accelerator.backward(scaled_fm_loss)

            # Extract head activations
            unwrapped_model = model.module if hasattr(model, 'module') else model
            head_acts = []
            for (layer_idx, head_idx) in selected_heads:
                o_proj = unwrapped_model.model.layers[layer_idx].self_attn.o_proj.to(torch.float32)
                if o_proj.bias is not None:
                    o_proj.bias.zero_()
                o_proj_input = forward_saved_inputs[layer_idx].to(torch.float32)
                if o_proj_input.ndim == 3:
                    o_proj_input = o_proj_input[:, -1, :]
                full_output = o_proj(o_proj_input)
                head_dim = unwrapped_model.config.hidden_size // unwrapped_model.config.num_attention_heads
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                head_acts.append(full_output[:, start:end])

            head_tensor = torch.stack(head_acts, dim=1)
            head_flat = head_tensor.view(head_tensor.size(0), -1)

            # Accumulate (gather from all processes)
            X_buffer.append(head_flat)
            Y_buffer.append((2 * labels - 1).view(-1))
            logits_buffer.append(logits)
            accumulated_samples += head_flat.size(0)

            # Update progress bar
            if accelerator.is_local_main_process:
                batch_progress.set_postfix({
                    'acc_samples': accumulated_samples,
                    'virtual_bs': args.virtual_batch_size,
                    'step': step
                })

            # Compute loss and step when virtual batch is ready
            if accumulated_samples >= args.virtual_batch_size:
                n = head_flat.shape[0]
                input_dim = head_flat.shape[1]

                Xc = torch.cat(X_buffer, dim=0)
                Yc = torch.cat(Y_buffer, dim=0)
                logits_c = torch.cat(logits_buffer, dim=0)
                Xc = Xc - Xc.mean(0, keepdim=True)

                # Gather from all processes for VAE training (only on main process)
                if accelerator.is_main_process:
                    # Gather data from all processes
                    gathered_X = accelerator.gather(Xc)
                    gathered_Y = accelerator.gather(Yc)
                    gathered_logits = accelerator.gather(logits_c)
                    
                    # Train VAE on main process only (to avoid duplication)
                    from utils_toxic import train_vae_and_extract_mu
                    train_mu, val_mu, c_all = train_vae_and_extract_mu(
                        gathered_X.clone().detach().cpu().numpy(),
                        gathered_Y.clone().detach().cpu().numpy(),
                        input_dim, z_dim=32, h_dim1=128, h_dim2=64,
                        batch_size=128, lr=1e-3, vae_epochs=100, 
                        dataset_name="toxigen_vicuna", 
                        model_name=args.model_name, device='cuda', args=args
                    )
                    # Use c_all for confounders
                    Cc = c_all[:len(gathered_X)].to(accelerator.device)
                    # Estimate sigma_sq from reconstruction error
                    sigma_sq = torch.var(gathered_X - gathered_X.mean(0, keepdim=True)).item()
                else:
                    Cc = torch.zeros((len(Xc), 32), device=accelerator.device)
                    sigma_sq = 1.0
                
                # Broadcast confounders to all processes
                Cc = accelerator.broadcast(Cc, from_process=0)
                sigma_sq = accelerator.broadcast(torch.tensor(sigma_sq, device=accelerator.device), from_process=0).item()
                
                # Compute main loss (logPNS)
                XtX = Xc.T @ Xc
                XtY = Xc.T @ Yc.to(Xc.dtype)
                
                XtX_f = XtX.clone().detach().to(torch.float32)
                XtY_f = XtY.clone().detach().to(torch.float32)
                I = torch.eye(Xc.shape[1], device=accelerator.device, dtype=torch.float32)
                beta = torch.linalg.solve(XtX_f + args.lambda_reg * I, XtY_f.unsqueeze(1))
                
                CtC = Cc.T @ Cc
                CtY = Cc.T @ Yc.to(Cc.dtype)
                CtC_f = CtC.to(torch.float32)
                CtY_f = CtY.to(torch.float32)
                I2 = torch.eye(Cc.shape[1], device=accelerator.device, dtype=torch.float32)
                gamma = torch.linalg.solve(CtC_f + args.lambda_reg * I2, CtY_f.unsqueeze(1))
                
                beta = beta.to(Xc.dtype)
                gamma = gamma.to(Cc.dtype)
                proj = torch.matmul(Xc, beta)
                conf = Cc @ gamma
                conf_adj = proj * conf
                term1 = (proj**2).sum()
                term2 = 2 * conf_adj.sum()
                logpns = (term1 + args.lambda_term2 * term2) / (2 * sigma_sq)
                
                total_loss = logpns
                
                # Add L2 regularization if enabled
                if args.use_l2:
                    unwrapped_model = model.module if hasattr(model, 'module') else model
                    l2_loss = calculate_l2_loss(unwrapped_model, selected_heads, args.l2_lambda)
                    total_loss = total_loss + l2_loss
                
                loss = total_loss

                # Backward pass
                accelerator.backward(loss)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                step += 1
                if accelerator.is_local_main_process:
                    progress_bar.set_description(f"Step {step}")
                    progress_bar.set_postfix(loss=loss.detach().item())
                    progress_bar.update(1)
                    if isinstance(batch_progress, tqdm):
                        batch_progress.set_postfix({
                            'acc_samples': accumulated_samples,
                            'virtual_bs': args.virtual_batch_size,
                            'step': step,
                            'loss': f"{loss.detach().item():.4f}"
                        })

                # Reset buffers
                X_buffer.clear()
                Y_buffer.clear()
                C_buffer.clear()
                logits_buffer.clear()
                accumulated_samples = 0

                if step % 50 == 0 and accelerator.is_local_main_process:
                    print("Training loss:", loss.detach().item())
                    gc.collect()


def main():
    args = parse_args()
    
    # Initialize Accelerator with FSDP for memory efficiency
    # FSDP shards the model across GPUs instead of replicating it (like DDP)
    # This is essential for large models like vicuna_13B on A100 40GB
    accelerator = Accelerator(
        mixed_precision="fp16",  # Use mixed precision for speed
        gradient_accumulation_steps=1,  # We handle this manually with virtual_batch_size
        # Note: FSDP configuration is typically done via accelerate config
        # Run: accelerate config and select FSDP option
        # For now, we'll use DDP but with optimizations
    )
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    device = accelerator.device
    
    if accelerator.is_local_main_process:
        print(f"🚀 Training on {accelerator.num_processes} GPUs/processes")
        print(f"   Device: {device}")
        print(f"   Mixed precision: fp16")
    
    # Load selected heads
    args.heads_path = f"/work/hdd/bcxt/yian3/toxic/features/heads/{args.use_pns}_{args.model_name}_{args.dataset_name}_seed_{args.seed}_top_{args.num_heads}_heads_fold_0.npy"
    if os.path.exists(args.heads_path):
        selected_heads = np.load(args.heads_path)
    else:
        args.heads_path = f"/work/hdd/bcxt/yian3/toxic/features/heads/{args.use_pns}_{args.model_name}_{args.dataset_name}_seed_{args.seed}_top_72_heads_fold_0.npy"
        selected_heads = np.load(args.heads_path)
    selected_heads = selected_heads[:args.num_heads] if len(selected_heads) > args.num_heads else selected_heads
    model_name = HF_NAMES[args.model_name]

    # Load model & tokenizer
    if accelerator.is_local_main_process:
        print("[DEBUG] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if accelerator.is_local_main_process:
        print("[DEBUG] Loading model on CPU (Accelerate will handle GPU placement)...", flush=True)
        print("   IMPORTANT: To avoid OOM, configure FSDP via 'accelerate config'", flush=True)
    
    # Load model on CPU - Accelerate will handle GPU placement
    # NOTE: If using DDP (default), each GPU needs full model copy (~26GB)
    # For vicuna_13B, this causes OOM. SOLUTION: Use FSDP via accelerate config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    # Ensure model is on CPU
    if next(model.parameters()).is_cuda:
        if accelerator.is_local_main_process:
            print("[DEBUG] Moving model from GPU to CPU...", flush=True)
        model = model.cpu()
        torch.cuda.empty_cache()
    
    if accelerator.is_local_main_process:
        model_device = next(model.parameters()).device
        print(f"[DEBUG] Model loaded on device: {model_device}", flush=True)
    
    if accelerator.is_local_main_process:
        print("[DEBUG] Configuring model...", flush=True)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup model parameters
    if accelerator.is_local_main_process:
        print("[DEBUG] Setting up model parameters...")
    params_to_update, forward_saved_inputs = setup_model_parameters(model, selected_heads, args)
    
    if accelerator.is_local_main_process:
        print(f"[DEBUG] Created optimizer with {len(params_to_update)} parameter groups", flush=True)
    optimizer = torch.optim.AdamW(params_to_update, lr=args.lr, maximize=True, weight_decay=0.01)

    # Setup dataset & dataloaders
    if accelerator.is_local_main_process:
        print("[DEBUG] Loading dataset...", flush=True)
    dataset = ToxicDataset_paired(
        f"/work/hdd/bcxt/yian3/toxic/features/{args.dataset_name}_texts.json", 
        tokenizer, 
        max_len=args.max_length
    )
    
    if accelerator.is_local_main_process:
        print(f"[DEBUG] Dataset loaded: {len(dataset)} samples", flush=True)
        print("[DEBUG] Setting up dataloaders...", flush=True)
    train_loader, eval_loader = setup_dataloaders(dataset, args)
    
    if accelerator.is_local_main_process:
        print(f"[DEBUG] Train loader: {len(train_loader)} batches", flush=True)
        print("[DEBUG] Preparing model/optimizer/dataloaders with accelerator...", flush=True)
        print("   (This may take a moment for multi-GPU setup...)", flush=True)
        print("   WARNING: If using DDP (default), ensure FSDP is configured!", flush=True)
        print("   Run 'accelerate config' and select FSDP to avoid OOM.", flush=True)
        
        # Clear GPU cache before prepare() to avoid OOM
        torch.cuda.empty_cache()
    
    # Prepare model, optimizer, dataloaders with accelerator
    # NOTE: If using DDP (default), each GPU needs full model copy (~26GB)
    # For vicuna_13B on 4x A100 40GB, this can cause OOM
    # SOLUTION: Configure FSDP via 'accelerate config' to shard model across GPUs
    try:
        model, optimizer, train_loader, eval_loader = accelerator.prepare(
            model, optimizer, train_loader, eval_loader
        )
    except torch.cuda.OutOfMemoryError as e:
        if accelerator.is_local_main_process:
            print("\n" + "="*80, flush=True)
            print("ERROR: CUDA Out of Memory during accelerator.prepare()", flush=True)
            print("="*80, flush=True)
            print("SOLUTION: Configure Accelerate to use FSDP (Fully Sharded Data Parallel):", flush=True)
            print("  1. Run: accelerate config", flush=True)
            print("  2. Select: 'Fully Sharded Data Parallel (FSDP)'", flush=True)
            print("  3. This will shard the model across GPUs instead of replicating it", flush=True)
            print("  4. Memory usage: ~26GB total → ~6.5GB per GPU (for 4 GPUs)", flush=True)
            print("="*80 + "\n", flush=True)
        raise
    
    if accelerator.is_local_main_process:
        print("[DEBUG] Accelerator preparation complete!", flush=True)

    # Setup teacher model (only on main process to save memory)
    teacher_model = None
    if args.use_kl:
        if accelerator.is_main_process:
            if accelerator.is_local_main_process:
                print("[DEBUG] Creating teacher model (on main process only, loading from disk)...")
            # Load teacher model from disk (original pretrained model)
            teacher_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu"  # Keep on CPU to save GPU memory
            ).eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)
            if accelerator.is_local_main_process:
                print("[DEBUG] Teacher model loaded and frozen")
        else:
            teacher_model = None

    # Train loop
    if accelerator.is_local_main_process:
        print("[DEBUG] Starting training loop...")
    train_model(
        accelerator=accelerator,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        selected_heads=selected_heads,
        params_to_update=params_to_update,
        forward_saved_inputs=forward_saved_inputs,
        args=args,
        tokenizer=tokenizer,
        teacher_model=teacher_model
    )

    # Save model (only on main process)
    if accelerator.is_main_process:
        save_dir = f"/work/hdd/bcxt/yian3/toxic/models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.generation_config.temperature = None
        unwrapped_model.generation_config.top_p = None
        unwrapped_model.generation_config.top_k = None
        unwrapped_model.generation_config.do_sample = True
        
        if 'True' in args.heads_path:
            use_pns_head = 'True'
        else:
            use_pns_head = 'False'
        
        l2_suffix = f"_l2{args.l2_lambda}" if args.use_l2 else ""
        output_dir = f"/work/hdd/bcxt/yian3/toxic/models/{args.model_name}_{args.dataset_name}_{args.head_select}_{args.num_heads}_{use_pns_head}_{args.lr}_{args.lambda_term2}_finetuned{l2_suffix}_useKL_{args.use_kl}_{args.lambda_fm}_epoch{args.epochs}_accelerate"
        
        unwrapped_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"\n✅ {args.model_name} fine-tuning complete. Saved to {output_dir}")


if __name__ == "__main__":
    main()

