# Register Forward Hooks to Extract Head Activations
head_outputs = {}

def make_hook(layer_idx, head_idx):
    def hook(module, input, output):
        # output shape: [batch, seq_len, n_heads * head_dim]
        batch, seq, dim = output.shape
        head_dim = dim // module.num_heads
        # reshape: [batch, seq_len, num_heads, head_dim]
        output = output.view(batch, seq, module.num_heads, head_dim)
        # take mean over sequence (or use last token)
        head_outputs[(layer_idx, head_idx)] = output[:, -1, head_idx, :]  # [batch, head_dim]
    return hook

def register_head_hooks(model, top_heads):
    for l, h in top_heads:
        attn = model.model.layers[l].self_attn
        attn.register_forward_hook(make_hook(l, h))

# Get Only the Parameters You Want to Update
def get_proj_params(model, top_heads):
    params = []
    seen = set()
    for l, h in top_heads:
        proj = model.model.layers[l].self_attn.o_proj
        if id(proj.weight) not in seen:
            proj.requires_grad_(True)
            params.append(proj.weight)
            seen.add(id(proj.weight))
    return params


# PNS Lower Bound Loss
def pns_lower_bound_loss(f_j, c_i, beta_j, gamma, sigma_sq=1.0):
    f_j_centered = f_j - f_j.mean(dim=0, keepdim=True)
    c_centered = c_i - c_i.mean(dim=0, keepdim=True)
    linear_term = (f_j_centered @ beta_j)  # [B]
    interaction_term = linear_term * (c_centered @ gamma)  # [B]
    loss = - (1 / (2 * sigma_sq)) * (linear_term ** 2 + 2 * interaction_term).mean()
    return loss

# Training Loop
def fine_tune_heads(model, dataloader, top_heads, beta_dict, gamma_dict, device, lr=1e-4, epochs=2):
    model.to(device)
    model.train()
    
    register_head_hooks(model, top_heads)
    params = get_proj_params(model, top_heads)
    optimizer = torch.optim.Adam(params, lr=lr)
    
    for epoch in range(epochs):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            causes = batch['causes'].to(device)  # shape [B, d_c]

            head_outputs.clear()  # reset for this batch
            _ = model(input_ids, attention_mask=attention_mask)

            loss = 0.0
            for (l, h) in top_heads:
                f_j = head_outputs[(l, h)]  # [B, D]
                beta_j = beta_dict[(l, h)].to(device)
                gamma = gamma_dict[(l, h)].to(device)
                loss += pns_lower_bound_loss(f_j, causes, beta_j, gamma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
