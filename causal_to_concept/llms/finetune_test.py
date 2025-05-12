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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --- FREEZE & UNFREEZE SELECTED HEADS ---
params_to_update = []
head_dim = model.config.hidden_size // model.config.num_attention_heads


for param in model.parameters():
    param.requires_grad = False


for layer_idx, head_idx in selected_heads:
    o_proj = model.model.layers[layer_idx].self_attn.o_proj

    if o_proj.weight is not None:
        o_proj.weight.requires_grad = True
        params_to_update.append(o_proj.weight)

    if o_proj.bias is not None:
        o_proj.bias.requires_grad = True
        params_to_update.append(o_proj.bias)


optimizer = torch.optim.Adam(params_to_update, lr=lr)

# --- HOOK SETUP ---
collected_outputs = {}


@torch.no_grad()
def evaluate_model(model, dataloader, selected_heads):
    model.eval()
    print("\n🔍 Running evaluation on concatenated head outputs...")

    X_all = []
    y_all = []
    collected_outputs.clear()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)

        _ = model(input_ids=input_ids, attention_mask=attention_mask)

        # Collect and concatenate activations
        head_acts = [collected_outputs[(l, h)] for (l, h) in selected_heads]
        head_concat = torch.cat(head_acts, dim=-1).squeeze(0)  # [K * D_head]

        X_all.append(head_concat.cpu().numpy())
        y_all.append(label)

    X_all = np.stack(X_all)
    y_all = np.array(y_all)

    # Train-test split
    
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"✅ Eval after step:")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    model.train()



def make_hook(layer_idx, head_idx):
    def hook_fn(module, input, output):
        head_dim = output.shape[-1] // model.config.num_attention_heads
        start = head_idx * head_dim
        end = (head_idx + 1) * head_dim
        collected_outputs[(layer_idx, head_idx)] = output[:, -1, start:end]  # [B, D_head]
        print(f"[Hook] Layer {layer_idx}, Head {head_idx}, Output Shape: {output[:, -1, start:end].shape}")
    return hook_fn

# Register per (layer, head)
for (layer_idx, head_idx) in selected_heads:
    o_proj = model.model.layers[layer_idx].self_attn.o_proj
    o_proj.register_forward_hook(make_hook(layer_idx, head_idx))




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
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- TRAIN LOOP ---
step = 0
accumulated_samples = 0
virtual_batch_size = 64

for epoch in tqdm(range(epochs), desc="Epochs"):
    model.train()
    for batch in dataloader:
        collected_outputs.clear()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        batch_size_actual = input_ids.size(0)
        with autocast(dtype=torch.float16):  # Ensures grad compatibility with fp16
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # --- Collect only selected heads ---
            head_acts = []
            for (layer_idx, head_idx) in selected_heads:
                key = (layer_idx, head_idx)
                if key not in collected_outputs:
                    raise RuntimeError(f"Missing collected output for {key}. Hook likely not triggered.")
                head_acts.append(collected_outputs[key])

            head_tensor = torch.stack(head_acts, dim=1)  # [B, K, D_head]
            B, K, D = head_tensor.shape

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
            gamma = torch.linalg.solve(CtC + lambda_reg * I2, CtY).to(Cc.dtype)

            # --- projections ---
            proj = torch.einsum("bkd,kdl->bkl", Xc, beta).squeeze(-1)  # [B, K]
            conf = torch.matmul(Cc, gamma).squeeze(-1)
            conf_adj = proj * conf.unsqueeze(1)

            term1 = (proj**2).sum(0)
            term2 = 2 * conf_adj.sum(0)
            logpns = (term1 + term2) / (2 * sigma_sq)

            loss = -logpns.sum()

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
            accumulated_samples = 0  # reset counter

            # Optional: Eval after N steps
            if step % 100 == 0:
                evaluate_model(model, dataloader, selected_heads)


    print(f"Loss: {loss.item():.4f}")
    

print(f"\n✅ {model_name} log-PNS fine-tuning complete.")
