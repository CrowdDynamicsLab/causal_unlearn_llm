import torch
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
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
warnings.filterwarnings("ignore", message="`do_sample` is set to `False`")
warnings.filterwarnings("ignore", message="To copy construct from a tensor")
import argparse



# --- CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    'vicuna_13b': 'lmsys/vicuna-13b-v1.5',
}

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune language model with log-PNS objective')
    
    # Model and dataset configuration
    parser.add_argument('--model_name', type=str, default="vicuna_13B",
                      help='String identifier for the model')
    parser.add_argument('--dataset_name', type=str, default="toxigen_vicuna",
                      help='Name of the dataset')
    parser.add_argument('--toxic_path', type=str, default="./dataset/vicuna-13b_toxic.json",
                      help='Path to toxic dataset')
    parser.add_argument('--nontoxic_path', type=str, default="./dataset/vicuna-13b_nontoxic.json",
                      help='Path to non-toxic dataset')
    # Training hyperparameters
    parser.add_argument('--lambda_reg', type=float, default=1e-2,
                      help='Regularization strength')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--max_length', type=int, default=10,
                      help='Maximum sequence length')
    parser.add_argument('--virtual_batch_size', type=int, default=128,
                      help='Virtual batch size for gradient accumulation')
    
    # Model configuration
    parser.add_argument('--head_select', type=str, default='logpns',
                      help='Head selection method')
    parser.add_argument('--num_heads', type=int, default=10,
                      help='Number of heads to select')
    parser.add_argument('--heads_path', type=str, 
                      default="./False_vicuna_13B_toxigen_vicuna_seed_2_top_36_heads_alpha_5.0_fold_0_top_heads.npy",
                      help='Path to selected heads numpy file')
    
    parser.add_argument('--save_dir', type=str, 
                      default="/projects/bdeb/chenyuen0103/toxic/models",
                      help='Directory to save model checkpoints')
    
    return parser.parse_args()


def train_vae_and_extract_mu(head_wise_activations, labels, input_dim, z_dim=1, h_dim1=128, h_dim2=64,
                              batch_size=128, lr=1e-3, vae_epochs=20, dataset_name=None, model_name=None, device='cuda'):
    split = int(0.8 * len(head_wise_activations))
    train_raw = head_wise_activations[:split]
    val_raw = head_wise_activations[split:]
    all_X_train = train_raw.cpu().to(torch.float32).view(-1, input_dim)
    all_X_val = val_raw.cpu().to(torch.float32).view(-1, input_dim)

    label_train_raw = labels[:split]
    label_val_raw = labels[split:]
    y_train = torch.tensor(label_train_raw, dtype=torch.float32)
    y_val = torch.tensor(label_val_raw, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(all_X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(all_X_val), batch_size=batch_size, shuffle=False)

    vae = VAE(input_dim, h_dim1, h_dim2, z_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(vae_epochs):
        vae.train()
        for (data,) in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var = vae(data)
            loss = vae_loss_function(recon_batch, data, mu, log_var)
            loss.backward()
            optimizer.step()

    # Estimate empirical sigma^2 (reconstruction variance)
    vae.eval()
    recon_errors = []
    with torch.no_grad():
        for (data,) in train_loader:
            data = data.to(device)
            recon_batch, _, _ = vae(data)
            recon_errors.append(((data - recon_batch) ** 2).sum(dim=1))  # shape: [batch]
    recon_errors = torch.cat(recon_errors, dim=0)
    sigma_sq_estimate = recon_errors.mean().item() / input_dim  # average over dimensions

    train_mu = get_all_mu(vae, all_X_train, batch_size=256, device=device)
    val_mu = get_all_mu(vae, all_X_val, batch_size=256, device=device)
    c_all = torch.cat([train_mu, val_mu], dim=0)

    acc, f1 = evaluate_latent_mu(train_mu.cpu(), y_train.cpu(), val_mu.cpu(), y_val.cpu())
    torch.save(c_all, f"/projects/bdeb/chenyuen0103/toxic/features/{args.model_name}_{args.dataset_name}_c_all.pt")

    return train_mu, val_mu, c_all, sigma_sq_estimate


forward_saved_inputs = {}


def make_hook(layer_idx):
    def hook_fn(module, input, output):
        x = input[0]
        forward_saved_inputs[layer_idx] = x
    return hook_fn

# Register hooks once per layer (not per head)
for layer_idx in set(l for (l, _) in selected_heads):
    o_proj = model.model.layers[layer_idx].self_attn.o_proj
    o_proj.register_forward_hook(make_hook(layer_idx))



# --- DATASET PREP ---
class ToxicDataset(Dataset):
    def __init__(self, toxic_path, nontoxic_path, tokenizer, max_len=64):
        toxic_data = load_dataset("json", data_files=toxic_path)["train"]
        nontoxic_data = load_dataset("json", data_files=nontoxic_path)["train"]

        texts = [x["toxic paraphrase"]["text"] for x in toxic_data] + \
                [x["non toxic paraphrase"]["text"] for x in nontoxic_data]

        labels = [1] * len(toxic_data) + [0] * len(nontoxic_data)


        self.encodings = tokenizer(texts, padding="max_length", truncation=True,
                                   max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": self.labels[idx],
            "index": torch.tensor(idx)  # <= new line
        }


class ToxicDataset_paired(Dataset):
    def __init__(self, toxic_path, tokenizer, max_len=64):
        toxic_data = load_dataset("json", data_files=toxic_path)["train"]  # Get the train split

        texts = []
        labels = []
        for item in toxic_data:
            if item["toxic_text"] != "(Blank)":  # Skip blank entries
                texts.append(item["toxic_text"])
                texts.append(item["non_toxic_text"])
                labels.extend([1, 0])  # 1 for toxic, 0 for non-toxic

        self.encodings = tokenizer(texts, padding="max_length", truncation=True,
                                   max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": self.labels[idx],
            "index": torch.tensor(idx)  # <= new line
        }


@torch.no_grad()
def evaluate_model(model, dataloader, selected_heads, lambda_reg=1e-2, sigma_sq=1.0):
    model.eval()
    device = next(model.parameters()).device

    all_acts = []
    all_labels = []
    all_confounds = []

    forward_saved_inputs.clear()

    for batch in dataloader:
        forward_saved_inputs.clear()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        indices = batch["index"].to(device)

        input_embeds = model.get_input_embeddings()(input_ids)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            _ = model(inputs_embeds=input_embeds, attention_mask=attention_mask)

        head_acts = []
        for (layer_idx, head_idx) in selected_heads:
            o_proj = model.model.layers[layer_idx].self_attn.o_proj
            o_proj_input = forward_saved_inputs[layer_idx]
            if o_proj_input.ndim == 3:
                o_proj_input = o_proj_input[:, -1, :]
            full_output = o_proj(o_proj_input.to(o_proj.weight.dtype))
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            head_acts.append(full_output[:, start:end])

        if len(head_acts) > 1:
            concat = torch.stack(head_acts, dim=1)  # [B, K, D_head]
        else:
            concat = head_acts[0].unsqueeze(1)

        all_acts.append(concat.cpu())
        all_labels.append(labels.cpu())
        # all_confounds.append(C[indices].cpu())

    # Stack results
    X_tensor = torch.cat(all_acts, dim=0).to(device)
    Y_tensor = torch.cat(all_labels, dim=0).to(device)
    # C_tensor = torch.cat(all_confounds, dim=0).to(device)

    # Center
    Xc = X_tensor - X_tensor.mean(0, keepdim=True)
    # Cc = C_tensor - C_tensor.mean(0, keepdim=True)
    Y_signed = (2 * Y_tensor.float() - 1)

    B, K, D = Xc.shape

    XtX = torch.einsum("bkd,bke->kde", Xc, Xc)
    XtY = torch.einsum("bkd,b->kd", Xc, Y_signed).unsqueeze(2)
    I = torch.eye(D, device=device, dtype=torch.float32).expand(K, D, D)
    beta = torch.linalg.solve(XtX.to(torch.float32) + lambda_reg * I, XtY.to(torch.float32)).to(Xc.dtype)

    # Confounder path
    # CtC = Cc.T @ Cc
    # CtY = Cc.T @ Y_signed
    # I2 = torch.eye(Cc.shape[1], device=device)
    # gamma = torch.linalg.solve(CtC + lambda_reg * I2, CtY.unsqueeze(1)).to(Cc.dtype)

    # Final loss terms
    proj = torch.einsum("bkd,kdl->bkl", Xc, beta).squeeze(-1)
    # conf = Cc @ gamma
    # conf_adj = proj * conf.unsqueeze(1)

    term1 = (proj**2).sum(0)
    # term2 = 2 * conf_adj.sum(0)
    # logpns = (term1 + term2).sum() / (2 * sigma_sq)
    logpns = term1.sum() / (2 * sigma_sq)

    # Classifier
    X_flat = X_tensor.view(B, -1).cpu().numpy()
    y_flat = Y_tensor.cpu().numpy()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_flat, y_flat)
    preds = clf.predict(X_flat)
    probs = clf.predict_proba(X_flat)[:, 1]

    print("\n📊 Evaluation:")
    print(f"   logPNS score: {logpns.item():.4f}")
    print(f"   Accuracy:     {accuracy_score(y_flat, preds):.4f}")
    print(f"   F1 Score:     {f1_score(y_flat, preds):.4f}")
    print(f"   ROC AUC:      {roc_auc_score(y_flat, probs):.4f}")

    # Cleanup
    del X_tensor, Y_tensor, proj, XtX, XtY, beta,
    torch.cuda.empty_cache()
    gc.collect()



dataset = ToxicDataset("./dataset/vicuna-13b_toxic.json", "./dataset/vicuna-13b_nontoxic.json", tokenizer, max_len=max_length)
# dataset = ToxicDataset_paired("./dataset/vicuna_13B_toxigen_vicuna_texts.json", tokenizer, max_len=max_length)
# breakpoint()

# Shuffle and split dataset indices
indices = list(range(len(dataset)))
np.random.seed(42)
np.random.shuffle(indices)

split_idx = int(0.8 * len(indices))
train_indices = indices[:split_idx]
eval_indices = indices[split_idx:]

train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(Subset(dataset, eval_indices), batch_size=8, shuffle=False)
# breakpoint()
# C_train = C 
# C_train = C[train_indices]
# C_eval = C[eval_indices]

# --- TRAIN LOOP ---
step = 0
accumulated_samples = 0
virtual_batch_size = 128

progress_bar = tqdm(total=epochs * ceil(len(train_loader) / (virtual_batch_size or 1)), desc="Steps")

X_buffer = []
Y_buffer = []
C_buffer = []





def setup_model_parameters(model, selected_heads):
    """Setup which parameters to update during training and register forward hooks."""
    params_to_update = []
    forward_saved_inputs = {}  # Dictionary to store inputs for each layer

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            x = input[0]
            forward_saved_inputs[layer_idx] = x
        return hook_fn

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Register hooks and unfreeze selected heads
    for layer_idx in set(l for (l, _) in selected_heads):
        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        # Register forward hook
        o_proj.register_forward_hook(make_hook(layer_idx))
        
        # Unfreeze parameters
        if o_proj.weight is not None:
            o_proj.weight.requires_grad = True
            params_to_update.append(o_proj.weight)
        if o_proj.bias is not None:
            o_proj.bias.requires_grad = True
            params_to_update.append(o_proj.bias)

    return params_to_update, forward_saved_inputs


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

def train_model(model, train_loader, eval_loader, optimizer, selected_heads, params_to_update, 
                device,forward_saved_inputs, args):
    """Main training loop."""
    step = 0
    accumulated_samples = 0
    progress_bar = tqdm(total=args.epochs * ceil(len(train_loader) / (args.virtual_batch_size or 1)), desc="Steps")

    X_buffer = []
    Y_buffer = []
    C_buffer = []

    for epoch in range(args.epochs):
        for param in params_to_update:
            if torch.isnan(param).any():
                print(f"NaNs in param {param.shape}")
                print(f"Require grad: {param.requires_grad}")

        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"🚨 NaN found at init in param: {name}, shape: {param.shape}")
        model.train()

        # breakpoint()   
        for batch in train_loader:
            forward_saved_inputs.clear()
            input_ids = batch['input_ids'].to(device)
            input_embeds = model.get_input_embeddings()(input_ids)
            # input_embeds.requires_grad = True
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            indices = batch["index"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _ = model(inputs_embeds=input_embeds, attention_mask=attention_mask)

                # --- Extract activations ---
            head_acts = []
            for (layer_idx, head_idx) in selected_heads:
                o_proj = model.model.layers[layer_idx].self_attn.o_proj.to(torch.float32)
                # o_proj.to(torch.float16)
                # if torch.isnan(o_proj.weight).any():
                #     print(f"🔁 Reinitializing o_proj at layer {layer_idx}")
                    # torch.nn.init.xavier_uniform_(o_proj.weight)
                if o_proj.bias is not None:
                    o_proj.bias.zero_()
                o_proj_input = forward_saved_inputs[layer_idx].to(torch.float32)
                if o_proj_input.ndim == 3:
                    o_proj_input = o_proj_input[:, -1, :]
                full_output = o_proj(o_proj_input)
                head_dim = model.config.hidden_size // model.config.num_attention_heads
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                head_acts.append(full_output[:, start:end])

            head_tensor = torch.stack(head_acts, dim=1)  # [B, K, D_head]
            head_flat = head_tensor.view(head_tensor.size(0), -1)  # [B, K*D_head]

            # Accumulate
            X_buffer.append(head_flat)  # Detach and clone to preserve the computation graph
            Y_buffer.append((2 * labels - 1).view(-1))
            # C_buffer.append(C_train[indices])
            accumulated_samples += head_flat.size(0)

            # Compute loss and step when virtual batch is ready
            if accumulated_samples >= virtual_batch_size:
                n = head_flat.shape[0]
                input_dim= head_flat.shape[1]
    

                Xc = torch.cat(X_buffer, dim=0)
                Yc = torch.cat(Y_buffer, dim=0)
                # Cc = torch.cat(C_buffer, dim=0)

                Xc = Xc - Xc.mean(0, keepdim=True)
                # Cc = Cc - Cc.mean(0, keepdim=True)


                XtX = Xc.T @ Xc
                XtY = Xc.T @ Yc.to(Xc.dtype)  # ensure both operands are float32

                XtX_f = XtX.clone().detach().to(torch.float32)
                XtY_f = XtY.clone().detach().to(torch.float32)
                I = torch.eye(Xc.shape[1], device=device,dtype = torch.float32)
                # breakpoint()
                beta = torch.linalg.solve(XtX_f + args.lambda_reg * I, XtY_f.unsqueeze(1))
                # beta = torch.linalg.lstsq(XtX_f, XtY_f.unsqueeze(1)).solution
                # beta = torch.ones(Xc.shape[1], 1, device=Xc.device, dtype=Xc.dtype) * 0.01
                # beta = torch.pinverse(XtX_f) @ XtY_f 

                head_wise_c = train_vae_and_extract_mu(Xc.clone().detach(), Yc.clone().detach(), input_dim, z_dim=32, h_dim1=128, h_dim2=64,
                            batch_size=128, lr=1e-3, vae_epochs=100, dataset_name="toxigen_vicuna", 
                            model_name=args.model_name, device='cuda')
                sigma_sq = head_wise_c[3]
                Cc = head_wise_c[2].detach().to(Xc.device)
                    


                CtC = Cc.T @ Cc
                CtY = Cc.T @ Yc.to(Cc.dtype)
                CtC_f = CtC.to(torch.float32)
                CtY_f = CtY.to(torch.float32)
                I2 = torch.eye(Cc.shape[1], device=device,dtype = torch.float32)
                gamma = torch.linalg.solve(CtC_f + args.lambda_reg * I2, CtY_f.unsqueeze(1))

                beta = beta.to(Xc.dtype)
                gamma = gamma.to(Cc.dtype)
                proj = torch.matmul(Xc, beta)  # [B, K, 1]
                conf = Cc @ gamma
                conf_adj = proj * conf
                term1 = (proj**2).sum()
                term2 = 2 * conf_adj.sum()
                # term2 = 0
                logpns = (term1 + term2) / (2 * sigma_sq)
                print("Term1:", term1)
                print("Term2:", term2)
                # logpns = term1/ (2*sigma_sq)
                # print("term1.requires_grad:", term1.requires_grad)
                # print("term2.requires_grad:", term2.requires_grad)


                loss = logpns
                loss.backward()
                optimizer.step()


                # for param in params_to_update:
                #     if torch.isnan(param).any():
                #         print(f"NaNs in param {param.shape}")
                #     if not param.grad.is_contiguous():
                #         print(f"Non-contiguous gradient of {param.shape}")
                optimizer.zero_grad()
                
                # for (layer_idx, head_idx) in selected_heads:
                #     o_proj = model.model.layers[layer_idx].self_attn.o_proj
                #     if torch.isnan(o_proj.weight).any():
                #         print(f"Nan in o_proj.weight of layer {layer_idx}")
                #     if o_proj.bias is not None:
                #         if torch.isnan(o_proj.bias).any():
                #             print(f"Nan in o_proj.bias of layer {layer_idx}")

                step += 1
                progress_bar.set_description(f"Step {step}")
                progress_bar.set_postfix(loss=loss.detach().item())
                progress_bar.update(1)

                # Reset
                X_buffer.clear()
                Y_buffer.clear()
                C_buffer.clear()
                accumulated_samples = 0

                if step % 50 == 0:
                    print("Training loss:", loss.detach().item())
                    # evaluate_model(model, eval_loader, selected_heads)
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Clear evaluation cache
                    
    progress_bar.close()


    # Save model

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = f"{args.save_dir}/{args.model_name}_{args.dataset_name}_logpns_finetuned_epoch{args.epochs}_lr{args.lr}_bs{args.virtual_batch_size}_lambda{args.lambda_reg}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ {args.model_name} log-PNS fine-tuning complete.")


def main():
    args = parse_args()
    
    # --- CONFIG ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load selected heads
    selected_heads = np.load(args.heads_path)
    selected_heads = selected_heads[:args.num_heads] if len(selected_heads) > args.num_heads else selected_heads

    # --- LOAD MODEL & TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16
    ).to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- SETUP MODEL PARAMETERS ---
    params_to_update, forward_saved_inputs = setup_model_parameters(model, selected_heads,args)
    optimizer = torch.optim.AdamW(params_to_update, lr=args.lr, maximize=True, weight_decay=0.01)

    # --- SETUP DATASET & DATALOADERS ---
    dataset = ToxicDataset(args.toxic_path, args.nontoxic_path, tokenizer, max_len=args.max_length)
    train_loader, eval_loader = setup_dataloaders(dataset, args)

    # --- TRAIN LOOP ---
    train_model(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        selected_heads=selected_heads,
        params_to_update=params_to_update,
        device=device,
        forward_saved_inputs=forward_saved_inputs,
        args=args
    )


    # save_dir = f"/projects/bdeb/chenyuen0103/toxic/models"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # save_path = f"{save_dir}/{args.model_name}_{args.dataset_name}_{args.head_select}_finetuned_epoch{args.epochs}_lr{args.lr}_bs{args.virtual_batch_size}_lambda{args.lambda_reg}.pt"
    # torch.save(model.state_dict(), save_path)
    # print(f"\n✅ {args.model_name} log-PNS fine-tuning complete.")





if __name__ == "__main__":
    main()