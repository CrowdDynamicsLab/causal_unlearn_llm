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

# --- CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name = "meta-llama/Meta-Llama-3-8B"
model_name = "lmsys/vicuna-13b-v1.5"
# model_name = "gpt2"
C = torch.load("/projects/bdeb/chenyuen0103/toxic/features/vicuna_13B_toxigen_vicuna_c_all.pt").to(device)


selected_heads = np.load("./False_vicuna_13B_toxigen_vicuna_seed_2_top_36_heads_alpha_5.0_fold_0_top_heads.npy")
selected_heads = selected_heads[:10]
lambda_reg = 1e-4
sigma_sq = 1.0
epochs = 2
batch_size = 1
lr = 1e-4
max_length = 64

# --- LOAD MODEL & TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)
model.gradient_checkpointing_enable()
model.config.use_cache = False  # Important when using gradient checkpointing

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --- FREEZE & UNFREEZE SELECTED HEADS ---
params_to_update = []
head_dim = model.config.hidden_size // model.config.num_attention_heads


for param in model.parameters():
    param.requires_grad = False


for layer_idx in set(l for (l, _) in selected_heads):
    o_proj = model.model.layers[layer_idx].self_attn.o_proj

    if o_proj.weight is not None:
        o_proj.weight.requires_grad = True
        params_to_update.append(o_proj.weight)

    if o_proj.bias is not None:
        o_proj.bias.requires_grad = True
        params_to_update.append(o_proj.bias)

# After your parameter setup code, add this debug check:
print("Checking parameter setup:")
for layer_idx, head_idx in selected_heads:
    o_proj = model.model.layers[layer_idx].self_attn.o_proj
    print(f"Layer {layer_idx}, Head {head_idx}:")
    print(f"  Weight requires_grad: {o_proj.weight.requires_grad}")
    print(f"  Weight grad_fn: {o_proj.weight.grad_fn}")
    if o_proj.bias is not None:
        print(f"  Bias requires_grad: {o_proj.bias.requires_grad}")
        print(f"  Bias grad_fn: {o_proj.bias.grad_fn}")

# After setting up parameters_to_update, verify the optimizer:
print("Optimizer parameters:")
for param in params_to_update:
    print(f"Parameter shape: {param.shape}, requires_grad: {param.requires_grad}")

optimizer = torch.optim.Adam(params_to_update, lr=lr)

# --- HOOK SETUP ---
collected_outputs = {}


@torch.no_grad()
def evaluate_model(model, dataloader, selected_heads):
    model.eval()
    forward_saved_inputs.clear()

    all_acts = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        input_embeds = model.get_input_embeddings()(input_ids)
        outputs = model(input_embeds=input_embeds, attention_mask=attention_mask)

        head_acts = []
        for (layer_idx, head_idx) in selected_heads:
            o_proj = model.model.layers[layer_idx].self_attn.o_proj
            o_proj_input = forward_saved_inputs[layer_idx]
            if o_proj_input.ndim == 3:
                o_proj_input = o_proj_input[:, -1, :]
            full_output = o_proj(o_proj_input)
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            head_acts.append(full_output[:, start:end])

        concat = torch.cat(head_acts, dim=-1)  # [B, K * D_head]
        all_acts.append(concat.cpu())
        all_labels.append(labels.cpu())

    X = torch.cat(all_acts).numpy()
    y = torch.cat(all_labels).numpy()

    # Train a quick linear classifier on eval set
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]

    print("✅ Evaluation (on held-out Theta subset):")
    print(f"Accuracy: {accuracy_score(y, preds):.4f} | "
          f"F1: {f1_score(y, preds):.4f} | "
          f"AUC: {roc_auc_score(y, probs):.4f}")


forward_saved_inputs = {}

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        # Save the input for this forward pass, don't store inside the module!
        forward_saved_inputs[layer_idx] = input[0]
        # print(f"[Hook] Layer {layer_idx} o_proj input shape: {input[0].shape}")
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
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": self.labels[idx],
            "index": torch.tensor(idx)  # <= new line
        }


dataset = ToxicDataset("./dataset/vicuna-13b_toxic.json", "./dataset/vicuna-13b_nontoxic.json", tokenizer, max_len=max_length)


# Shuffle and split dataset indices
indices = list(range(len(dataset)))
np.random.seed(42)
np.random.shuffle(indices)

split_idx = int(0.8 * len(indices))
train_indices = indices[:split_idx]
eval_indices = indices[split_idx:]

train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(Subset(dataset, eval_indices), batch_size=16, shuffle=False) 

# --- TRAIN LOOP ---
step = 0
accumulated_samples = 0
virtual_batch_size = 64

progress_bar = tqdm(total=epochs * ceil(len(train_loader) / (virtual_batch_size or 1)), desc="Steps")


for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        collected_outputs.clear()
        input_ids = batch['input_ids'].to(device)
        input_embeds = model.get_input_embeddings()(input_ids)
        input_embeds.requires_grad = True  # Force graph tracking
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        batch_size_actual = input_ids.size(0)
        with autocast(dtype=torch.float16):
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
            # Collect head activations
            head_acts = []
            for (layer_idx, head_idx) in selected_heads:
                o_proj = model.model.layers[layer_idx].self_attn.o_proj
                o_proj_input = forward_saved_inputs[layer_idx]  # Retrieved from forward hook
                
                # Handle both 2D and 3D input shapes
                if len(o_proj_input.shape) == 3:  # [B, T, D]
                    o_proj_input = o_proj_input[:, -1, :]  # [B, D]
                # If it's already 2D [B, D], use it as is
                
                full_output = o_proj(o_proj_input)     # [B, D]
                
                head_dim = model.config.hidden_size // model.config.num_attention_heads
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                head_acts.append(full_output[:, start:end])  # [B, D_head]

            # Stack the head activations
            head_tensor = torch.stack(head_acts, dim=1)  # [B, K, D_head]
            B = head_tensor.shape[0]
            K = head_tensor.shape[1]
            D = head_tensor.shape[2]
            
            # --- Confounders ---
            indices = batch["index"].to(device)
            c_batch = C[indices]

            if B > 1:
                Xc = head_tensor - head_tensor.mean(0, keepdim=True)
                Cc = c_batch - c_batch.mean(0, keepdim=True)
            else:
                Xc = head_tensor
                Cc = c_batch

            Y = 2 * labels - 1
            Y = Y.view(-1)

            # --- beta: head -> Y ---
            XtX = torch.einsum("bkd,bke->kde", Xc, Xc)
            XtY = torch.einsum("bkd,b->kd", Xc, Y).unsqueeze(2)
            I = torch.eye(D, device=device).expand(K, -1, -1)
            A = (XtX + lambda_reg * I).to(torch.float32)
            B_ = XtY.to(torch.float32)
            beta = torch.linalg.solve(A, B_).to(Xc.dtype)

            # --- gamma: confounder -> Y ---
            CtC = torch.einsum("bd,be->de", Cc, Cc)
            CtY = torch.einsum("bd,bl->dl", Cc, Y.unsqueeze(1))
            I2 = torch.eye(Cc.shape[1], device=device)
            A2 = (CtC + lambda_reg * I2).to(torch.float32)
            B2 = CtY.to(torch.float32)
            gamma = torch.linalg.solve(A2, B2).to(Cc.dtype)  # cast back to original dtype


            # --- projections ---
            proj = torch.einsum("bkd,kdl->bkl", Xc, beta).squeeze(-1)  # [B, K]
            conf = torch.matmul(Cc, gamma).squeeze(-1)
            conf_adj = proj * conf.unsqueeze(1)

            term1 = (proj**2).sum(0)
            term2 = 2 * conf_adj.sum(0)
            logpns = (term1 + term2) / (2 * sigma_sq)

            loss = -logpns.sum()

        # Check the data type of the loss tensor
        # print(f"Loss dtype: {loss.dtype}")
        # print(f"Loss requires_grad: {loss.requires_grad}")

        # # Check the data type of the model's parameters
        # print(f"Model parameter dtype: {model.parameters().__next__().dtype}")
        # print(f"Model parameter requires_grad: {model.parameters().__next__().requires_grad}")

        # # Check if the model is in training mode
        # print(f"Model training mode: {model.training}")

        assert loss.requires_grad, "Loss doesn't require grad, check dtype or computation path"
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad == False and param.grad is not None:
                    assert torch.all(param.grad == 0)


        accumulated_samples += batch_size_actual

        # Do optimizer step after reaching virtual batch
        if accumulated_samples >= virtual_batch_size:
            with torch.no_grad():
                for (layer_idx, head_idx) in selected_heads:
                    o_proj = model.model.layers[layer_idx].self_attn.o_proj
                    head_dim = model.config.hidden_size // model.config.num_attention_heads

                    for i in range(model.config.num_attention_heads):
                        if i == head_idx:
                            continue
                        start = i * head_dim
                        end = (i + 1) * head_dim
                        o_proj.weight.grad[:, start:end].zero_()
                        if o_proj.bias is not None:
                            o_proj.bias.grad[start:end].zero_()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            progress_bar.set_description(f"Step {step}")
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
            accumulated_samples = 0  # reset counter

            # Optional: Eval after N steps
            if step % 100 == 0:
                evaluate_model(model, eval_oader, selected_heads)
    progress_bar.close()
    

print(f"\n✅ {model_name} log-PNS fine-tuning complete.")

