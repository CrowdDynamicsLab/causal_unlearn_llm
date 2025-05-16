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
scaler = GradScaler()

# --- CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name = "meta-llama/Meta-Llama-3-8B"
model_name = "lmsys/vicuna-13b-v1.5"
# model_name = "gpt2"
C = torch.load("/work/hdd/bcxt/yian3/toxic/features/vicuna_13B_toxigen_vicuna_c_all.pt").to(device)


selected_heads = np.load("./features/False_vicuna_13B_toxigen_vicuna_seed_2_top_36_heads_alpha_5.0_fold_0_top_heads.npy")
selected_heads = selected_heads[:36]


# --- LOAD MODEL & TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
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


# Setup time only
for layer_idx in set(l for (l, _) in selected_heads):
    o_proj = model.model.layers[layer_idx].self_attn.o_proj
    if o_proj.weight is not None:
        o_proj.weight.requires_grad = True
        params_to_update.append(o_proj.weight)

    if o_proj.bias is not None:
        o_proj.bias.requires_grad = True
        params_to_update.append(o_proj.bias)

# After your parameter setup code, add this debug check:
# print("Checking parameter setup:")
# for layer_idx, head_idx in selected_heads:
#     o_proj = model.model.layers[layer_idx].self_attn.o_proj
#     print(f"Layer {layer_idx}, Head {head_idx}:")
#     print(f"  Weight requires_grad: {o_proj.weight.requires_grad}")
#     print(f"  Weight grad_fn: {o_proj.weight.grad_fn}")
#     if o_proj.bias is not None:
#         print(f"  Bias requires_grad: {o_proj.bias.requires_grad}")
#         print(f"  Bias grad_fn: {o_proj.bias.grad_fn}")

# After setting up parameters_to_update, verify the optimizer:
print("Optimizer parameters:")
for param in params_to_update:
    if not param.requires_grad:
        print(f"Parameter shape: {param.shape}, requires_grad: {param.requires_grad}")

lambda_reg = 1e-4  # 1e-2
sigma_sq = 1.0
epochs = 5
batch_size = 1
lr = 1e-3
max_length = 64
optimizer = torch.optim.AdamW(params_to_update, lr=lr, weight_decay=5e-4)

# --- HOOK SETUP ---
collected_outputs = {}


@torch.no_grad()
def evaluate_model(model, dataloader, selected_heads):
    model.eval()
    forward_saved_inputs.clear()
    
    # Temporarily disable gradient checkpointing during evaluation
    was_checkpointing = not model.config.use_cache  # Check if checkpointing is enabled
    if was_checkpointing:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
    
    try:
        all_acts = []
        all_labels = []
        all_confounds = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            indices = batch["index"].to(device)

            input_embeds = model.get_input_embeddings()(input_ids)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

            head_acts = []
            for (layer_idx, head_idx) in selected_heads:
                o_proj = model.model.layers[layer_idx].self_attn.o_proj.to(torch.float32)
                # check for nan in o_proj.weight and o_proj.bias


                o_proj_input = forward_saved_inputs[layer_idx].to(torch.float32)
                if o_proj_input.ndim == 3:
                    o_proj_input = o_proj_input[:, -1, :]  # take last token
                full_output = o_proj(o_proj_input)
                head_dim = model.config.hidden_size // model.config.num_attention_heads
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                head_acts.append(full_output[:, start:end])

            concat = torch.stack(head_acts, dim=1)  # [B, K, D_head]
            all_acts.append(concat.cpu())
            all_labels.append(labels.cpu())
            all_confounds.append(C[indices].cpu())

        # Stack all batches
        X_tensor = torch.cat(all_acts, dim=0).to(device)               # [N, K, D_head]
        Y = torch.cat(all_labels, dim=0).to(device)                    # [N]
        C_tensor = torch.cat(all_confounds, dim=0).to(device)          # [N, D_c]

        B, K, D = X_tensor.shape

        # --- Centered versions ---
        Xc = X_tensor - X_tensor.mean(0, keepdim=True) if X_tensor.shape[0] > 1 else X_tensor
        Cc = C_tensor - C_tensor.mean(0, keepdim=True) if C_tensor.shape[0] > 1 else C_tensor
        Y_signed = 2 * Y - 1
        Y_signed = Y_signed.view(-1)

        # Cast to float32 before einsum
        Xc_f = Xc
        Y_signed_f = Y_signed

        XtX = torch.einsum("bkd,bke->kde", Xc_f, Xc_f)
        XtY = torch.einsum("bkd,b->kd", Xc_f, Y_signed_f).unsqueeze(2)

        I = torch.eye(D, device=device).expand(K, -1, -1)
        A = (XtX + lambda_reg * I)
        B_ = XtY
        XtX_f = XtX.to(torch.float32)
        XtY_f = XtY.to(torch.float32)
        beta = torch.linalg.solve(XtX_f + lambda_reg * I, XtY_f.unsqueeze(1))

        # --- gamma: confounder -> Y ---
        CtC = torch.einsum("bd,be->de", Cc, Cc)
        CtY = torch.einsum("bd,b->d", Cc, Y_signed).unsqueeze(1)
        I2 = torch.eye(Cc.shape[1], device=device)
        A2 = (CtC + lambda_reg * I2)
        B2 = CtY
        gamma = torch.linalg.solve(A2, B2).to(Cc.dtype)

        # --- projection and adjustment ---
        proj = torch.einsum("bkd,kdl->bkl", Xc, beta).squeeze(-1)  # [B, K]
        conf = torch.matmul(Cc, gamma).squeeze(-1)
        conf_adj = proj * conf.unsqueeze(1)

        term1 = (proj**2).sum(0)
        # term2 = 2 * conf_adj.sum(0)
        logpns_score = (term1).sum() / (2 * sigma_sq)

        # --- Classifier for accuracy/F1/AUC ---
        X_flat = X_tensor.view(B, -1).cpu().numpy()
        y_flat = Y.cpu().numpy()
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_flat, y_flat)
        preds = clf.predict(X_flat)
        probs = clf.predict_proba(X_flat)[:, 1]

        acc = accuracy_score(y_flat, preds)
        f1 = f1_score(y_flat, preds)
        auc = roc_auc_score(y_flat, probs)

        print("\n✅ Evaluation on held-out set:")
        print(f"logPNS Score: {logpns_score.item():.4f}")
        print(f"Accuracy:     {acc:.4f}")
        print(f"F1 Score:     {f1:.4f}")
        print(f"ROC AUC:      {auc:.4f}")

    finally:
        # Restore previous gradient checkpointing state
        if was_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False


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


dataset = ToxicDataset("../dataset/vicuna-13b_toxic.json", "../dataset/vicuna-13b_nontoxic.json", tokenizer, max_len=max_length)


# Shuffle and split dataset indices
indices = list(range(len(dataset)))
np.random.seed(42)
np.random.shuffle(indices)

split_idx = int(0.8 * len(indices))
train_indices = indices[:split_idx]
eval_indices = indices[split_idx:]

train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(Subset(dataset, eval_indices), batch_size=64, shuffle=False) 

# --- TRAIN LOOP ---
step = 0
accumulated_samples = 0
virtual_batch_size = 128

progress_bar = tqdm(total=epochs * ceil(len(train_loader) / (virtual_batch_size or 1)), desc="Steps")

X_buffer = []
Y_buffer = []
C_buffer = []

for epoch in range(epochs):
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
        collected_outputs.clear()
        input_ids = batch['input_ids'].to(device)
        input_embeds = model.get_input_embeddings()(input_ids)
        input_embeds.requires_grad = True
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        indices = batch["index"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):

            _ = model(inputs_embeds=input_embeds, attention_mask=attention_mask)

        # --- Extract activations ---
        head_acts = []
        for (layer_idx, head_idx) in selected_heads:
            o_proj = model.model.layers[layer_idx].self_attn.o_proj.to(torch.float32)
            # o_proj.to(torch.float32)
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
        C_buffer.append(C[indices])


        accumulated_samples += head_flat.size(0)

        # Compute loss and step when virtual batch is ready
        if accumulated_samples >= virtual_batch_size:
            Xc = torch.cat(X_buffer, dim=0)
            Yc = torch.cat(Y_buffer, dim=0)
            Cc = torch.cat(C_buffer, dim=0)

            Xc = Xc - Xc.mean(0, keepdim=True)
            Cc = Cc - Cc.mean(0, keepdim=True)

            XtX = Xc.T @ Xc
            XtY = Xc.T @ Yc.to(Xc.dtype)  # ensure both operands are float32

            XtX_f = XtX.to(torch.float32)
            XtY_f = XtY.to(torch.float32)
            I = torch.eye(Xc.shape[1], device=device,dtype = torch.float32)
            # breakpoint()
            beta = torch.linalg.solve(XtX_f + lambda_reg * I, XtY_f.unsqueeze(1))
            # beta = torch.pinverse(XtX_f) @ XtY_f 




            CtC = Cc.T @ Cc
            CtY = Cc.T @ Yc.to(Cc.dtype)
            CtC_f = CtC.to(torch.float32)
            CtY_f = CtY.to(torch.float32)
            I2 = torch.eye(Cc.shape[1], device=device,dtype = torch.float32)
            gamma = torch.linalg.solve(CtC_f + lambda_reg * I2, CtY_f.unsqueeze(1))
            # gamma = torch.pinverse(CtC_f) @ CtY_f

            beta = beta.to(Xc.dtype)
            gamma = gamma.to(Cc.dtype)


            proj = Xc @ beta
            conf = Cc @ gamma
            conf_adj = proj * conf
            term1 = (proj**2).sum()
            # term2 = 2 * conf_adj.sum()
            logpns = (term1) / (2 * sigma_sq) #  + term2
            print("term1.requires_grad:", term1.requires_grad)
            # print("term2.requires_grad:", term2.requires_grad)


            loss = -logpns
            print("loss.requires_grad:", loss.requires_grad)
            print("loss.grad_fn:", loss.grad_fn)
            # breakpoint()
            loss.backward()
            print("Loss:", loss.detach().item())
            print("proj std:", proj.std().item())
            print("beta norm:", beta.norm().item())
            print("Max grad val:", max(param.grad.abs().max().item() for param in model.parameters() if param.grad is not None))
            for param in params_to_update:
                if torch.isnan(param).any():
                    print(f"NaNs in param {param.shape}")
                if not param.grad.is_contiguous():
                    print(f"Non-contiguous gradient of {param.shape}")
            
            optimizer.step()
            
            # scaler.step(optimizer)
            # scaler.update()

            for param in params_to_update:
                if torch.isnan(param).any():
                    print(f"NaNs in param {param.shape}")
                if not param.grad.is_contiguous():
                    print(f"Non-contiguous gradient of {param.shape}")
            optimizer.zero_grad()
            
            # for (layer_idx, head_idx) in selected_heads:

            #     o_proj = model.model.layers[layer_idx].self_attn.o_proj.to(torch.float32)
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

            # if step % 2 == 0:
            #     evaluate_model(model, eval_loader, selected_heads)
    progress_bar.close()

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None  # if set
    model.generation_config.do_sample = True  # to match temp/top_p
    output_dir = "/work/hdd/bcxt/yian3/models/vicuna_pns_finetuned"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

print(f"\n✅ {model_name} log-PNS fine-tuning complete.")

