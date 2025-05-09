import numpy as np
from sklearn.model_selection import train_test_split
from utils_toxic import get_top_heads, get_separated_activations

# Load
X = np.load("/projects/bdeb/chenyuen0103/toxic/features/vicuna_13B_toxigen_vicuna_head_wise.npy")  # (samples, layers, heads, dim)
y = np.load("/projects/bdeb/chenyuen0103/toxic/features/vicuna_13B_toxigen_vicuna_labels.npy")     # (samples,)
# head_wise_activations = np.rearrange(X, 'b l (h d) -> b l h d', h = 40)
num_samples, num_layers, num_heads, dim  = X.shape
# Split into training/validation indices
train_idx, val_idx = train_test_split(np.arange(num_samples), test_size=0.2, random_state=42)

# Reshape to mimic `separated_head_wise_activations = [X[i:i+1] for i in range(len(X))]`
separated_activations = [X[i:i+1] for i in range(len(X))]
separated_labels = [y[i:i+1] for i in range(len(y))]

# Get top K heads
top_heads, trained_probes = get_top_heads(
    train_idx,
    val_idx,
    separated_activations,
    separated_labels,
    num_layers=num_layers,
    num_heads=num_heads,
    seed=42,
    num_to_intervene=10,
    use_random_dir=False
)

print("Top heads:", top_heads)
