from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils_toxic import (flattened_idx_to_layer_head)
from tqdm import tqdm
from einops import rearrange
from datasets import load_dataset
import os
import pyvene as pv
from get_activations.interveners import wrapper,ITI_Intervener, Collector


class Collector:
    def __init__(self, head=-1, num_heads=None):
        self.head = head
        self.num_heads = num_heads
        self.output = None

    def reset(self):
        self.output = None

    def __call__(self, b, s):
        # b: [B, T, D]
        final_token = b[:, -1]  # [B, D]
        if self.head == -1 or self.num_heads is None:
            self.output = final_token
        else:
            d = final_token.shape[-1]
            head_dim = d // self.num_heads
            reshaped = final_token.view(final_token.size(0), self.num_heads, head_dim)  # [B, H, D//H]
            self.output = reshaped[:, self.head, :]  # [B, D//H]
        return b  # very important to return original `b`



model_name = "lmsys/vicuna-13b-v1.5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

top_k_heads = np.load("./False_vicuna_13B_toxigen_vicuna_seed_2_top_36_heads_alpha_5.0_fold_0_top_heads.npy")

C = torch.load("/projects/bdeb/chenyuen0103/toxic/features/vicuna_13B_toxigen_vicuna_c_all.pt").to(device)




# Load toxic and non-toxic datasets
toxic_set = load_dataset("json", data_files="./dataset/vicuna-13b_toxic.json")["train"]
nontoxic_set = load_dataset("json", data_files="./dataset/vicuna-13b_nontoxic.json")["train"]

# Combine and label
# breakpoint()
texts = [x["toxic paraphrase"]['text'] for x in toxic_set] + [x["non toxic paraphrase"]['text'] for x in nontoxic_set]
labels = [1] * len(toxic_set) + [0] * len(nontoxic_set)
N = len(texts)


# Validate texts before creating dataset
valid_texts = []
valid_labels = []
for i, (text, label) in enumerate(zip(texts, labels)):
    if isinstance(text, str):
        valid_texts.append(text)
        valid_labels.append(label)
    else:
        print(f"Skipping invalid text at index {i}: {type(text)}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.to(device)
H = model.config.num_attention_heads
L = model.config.num_hidden_layers
D = model.config.hidden_size


C_dim = C.shape[1]

# ---------- Freeze all weights ----------
for param in model.parameters():
    param.requires_grad = False


# ---------- Unfreeze selected heads ----------
head_slices = {}
for layer_idx, head_idx in top_k_heads:
    proj = model.model.layers[layer_idx].self_attn.o_proj
    head_dim = proj.out_features // model.config.num_attention_heads
    start, end = head_idx * head_dim, (head_idx + 1) * head_dim

    proj.weight.requires_grad = True
    if proj.bias is not None:
        proj.bias.requires_grad = True

    head_slices[(layer_idx, head_idx)] = (proj, start, end)

# ---------- Optimizer ----------
params_to_update = []
for proj, _, _ in head_slices.values():
    params_to_update.append(proj.weight)
    if proj.bias is not None:
        params_to_update.append(proj.bias)

optimizer = torch.optim.Adam(params_to_update, lr=1e-4)

# ---------- Train Loop ----------
epochs = 5
batch_size = 1
lambda_reg = 1e-4
sigma_sq = 1.0




def compute_multihead_pns(
    all_heads_out,      # [B, K, D_head]  fp16 (or fp32)
    Y_batch,            # [B, 1]         same dtype
    C_batch,            # [B, C]         same dtype (unused here)
    lambda_reg=1e-4,
    sigma_sq=1.0,
):
    """
    Differentiable log‑PNS for K heads.
    Works on fp16 inputs by up‑casting only the small solve matrices.
    Returns: tensor [K]  (same dtype as inputs)
    """
    dtype  = all_heads_out.dtype
    device = all_heads_out.device
    B, K, D = all_heads_out.shape

    # --- center X and Y -------------------------------------------------
    Xc = all_heads_out - all_heads_out.mean(0, keepdim=True)       # [B,K,D]
    Y  = Y_batch.unsqueeze(1)                                      # [B,1,1]

    # --- build XtX and XtY --------------------------------------------
    XtX = torch.einsum("bkd,bke->kde", Xc, Xc)                     # [K,D,D]
    XtY = (Xc * Y).sum(0).unsqueeze(-1)                            # [K,D,1]

    # --- ridge term ----------------------------------------------------
    I   = torch.eye(D, dtype=dtype, device=device).expand(K, -1, -1)

    # --- cast the tiny blocks to fp32, solve, cast back ----------------
    A32 = (XtX + lambda_reg * I).to(torch.float32)                 # [K,D,D]
    B32 = XtY.to(torch.float32)                                    # [K,D,1]
    beta = torch.linalg.solve(A32, B32).to(dtype)                  # [K,D,1]

    # --- project & compute log‑PNS ------------------------------------
    proj  = torch.einsum("bkd,kde->bke", Xc, beta).squeeze(-1)     # [B,K]
    term1 = (proj**2).sum(0)                                       # [K]

    return term1 / (2 * sigma_sq)                                  # [K]

# --- Corrected Collector Setup ---
# Create one collector for the o_proj.input of each unique layer in top_k_heads
layer_to_oproj_input_collector = {}
pv_config = []
# Get unique layer indices from top_k_heads
unique_layer_indices = sorted(list(set(lh[0] for lh in top_k_heads)))

for layer_idx in unique_layer_indices:
    # This collector will grab the full [B, D_model] input to o_proj
    collector = Collector(
        head=-1,  # head=-1 tells the Collector to store the full tensor
        num_heads=None # Not needed when head=-1
    )
    layer_to_oproj_input_collector[layer_idx] = collector
    pv_config.append({
        "component": f"model.layers[{layer_idx}].self_attn.o_proj.input",
        "intervention": wrapper(collector),
    })

# Wrap your model with pyvene using the new pv_config
collected_model = pv.IntervenableModel(pv_config, model)


class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels)
        
        # Add a check to ensure encodings and labels have the same length
        if len(self.encodings['input_ids']) != len(self.labels):
            raise ValueError(f"Mismatch between number of encodings ({len(self.encodings['input_ids'])}) and labels ({len(self.labels)})")
        
        print(f"Dataset initialized with {len(self.labels)} samples")  # Add debug print

    def __len__(self):
        return len(self.labels)  # This method is required for DataLoader

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


dataset = ToxicDataset(valid_texts, valid_labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    for batch in tqdm(dataloader):
        x_batch, y_batch = batch["input_ids"], batch["labels"]
        c_batch = C[x_batch[:, 0]]
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        c_batch = c_batch.to(device)
        inputs = {
            "input_ids": x_batch,
            "attention_mask": batch["attention_mask"].to(device),
        }

        # 1. Reset collectors before forward
        for collector in layer_to_oproj_input_collector.values():
            collector.reset()

        # 2. Forward pass to populate collectors (NO torch.no_grad() for training)
        with torch.enable_grad():                       # keep autograd on
            _ = collected_model(
                {"input_ids": x_batch,                 # <-- base dict
                "attention_mask": batch["attention_mask"].to(device),}                      # let Pyvene skip no_grad
            )
        # 3. Re-compute head activations differentiably
        head_acts_list = []
        for layer_idx, head_idx in top_k_heads:
            # Get the full [B, D_model] input to o_proj for this layer
            oproj_full_input = layer_to_oproj_input_collector[layer_idx].output
            
            if oproj_full_input is None:
                raise ValueError(f"Collector for o_proj.input of layer {layer_idx} did not collect data.")
            # Ensure batch dimension is present, even if batch_size is 1
            if oproj_full_input.dim() == 1 and x_batch.size(0) == 1: # If B=1 and collector squeezed
                 oproj_full_input = oproj_full_input.unsqueeze(0)
            if oproj_full_input.shape[0] != x_batch.size(0) or oproj_full_input.shape[1] != model.config.hidden_size:
                 raise ValueError(f"Shape mismatch for o_proj.input at layer {layer_idx}. Expected: [{x_batch.size(0)}, {model.config.hidden_size}], Got: {oproj_full_input.shape}")


            # Get the actual o_proj layer module
            o_proj_layer = model.model.layers[layer_idx].self_attn.o_proj

            # Re-compute the o_proj output using its collected input
            # oproj_full_input is [B, D_model], o_proj_layer is Linear(D_model, D_model)
            recomputed_oproj_full_output = o_proj_layer(oproj_full_input) # Shape: [B, D_model]

            # Now, slice the specific head from this recomputed full output
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            
            specific_head_activation = recomputed_oproj_full_output[:, start:end] # Shape: [B, D_head]
            head_acts_list.append(specific_head_activation)
        
        selected_head_activations = torch.stack(head_acts_list, dim=1)  # Shape: [B, K, D_head]

        # 4. Compute logpns loss (ensure selected_head_activations is on the correct device and dtype)
        logpns_scores = compute_multihead_pns(
            all_heads_out=selected_head_activations.to(device),
            Y_batch=y_batch.unsqueeze(1),
            C_batch=c_batch,
            lambda_reg=lambda_reg,
            sigma_sq=sigma_sq
        )

        total_loss = -logpns_scores.sum()

        optimizer.zero_grad()
        # Debug prints before backward
        print(f"total_loss: {total_loss.item()}, requires_grad: {total_loss.requires_grad}, grad_fn: {total_loss.grad_fn}")
        # Check grads of a parameter you expect to update
        # Example: print("Weight grad before backward:", list(head_slices.values())[0][0].weight.grad)

        total_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Loss: {total_loss.item():.4f}")

print("\n✅ Finetuning Complete.")

# Save the model
save_model_path = "/projects/bdeb/chenyuen0103/toxic/models"
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
torch.save(model.state_dict(), "/projects/bdeb/chenyuen0103/toxic/models/vicuna-13b-v1.5-pns-heads.bin")






