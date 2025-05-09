from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

model_name = "lmsys/vicuna-13b-v1.5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


top_k_heads = np.load("./False_vicuna_13B_toxigen_vicuna_seed_2_top_36_heads_alpha_5.0_fold_0_top_heads.npy")

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the top attention heads
for (layer_idx, head_idx) in top_k_heads:
    # Locate the output projection matrix of each head
    attn_proj = model.model.layers[layer_idx].self_attn.o_proj

    # Unfreeze corresponding head slice
    head_dim = attn_proj.out_features // model.config.num_attention_heads
    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim

    # Enable gradient only on relevant slice of the projection weight
    attn_proj.weight.requires_grad = True
    attn_proj.bias.requires_grad = True

    # Optional: mask out gradient outside the selected slice
    # Could be done using hooks or during optimizer step
