import json
import csv
from datasets import load_dataset
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D
from utils_toxic import *
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


num_heads = 32
act =  np.load(f'/work/hdd/bcxt/yian3/toxic/features/llama_7B_toxigen_vicuna_head_wise.npy')
head_wise_activations = rearrange(act, 'b l (h d) -> b l h d', h = num_heads)
# act_layer =  np.load(f'/work/hdd/bcxt/yian3/toxic/features/llama_7B_toxigen_vicuna_layer_wise.npy')
top_heads = np.load(f'./features/llama_7B_toxigen_vicuna_top_heads.npy')
labels =  np.load(f'/work/hdd/bcxt/yian3/toxic/features/llama_7B_toxigen_vicuna_labels.npy')
dataset_name = 'toxigen_vicuna'
model_name = 'llama_7B'
separated_head_wise_activations, separated_labels, idxs_to_split_at = get_activations(labels, head_wise_activations, dataset_name, model_name)
# all_activations = np.concatenate(separated_head_wise_activations, axis=0)
selected = []
for sample in tqdm(head_wise_activations[:10000]):  # sample: [32, 32, 128]
    head_vectors = []
    for layer, head in top_heads:
        vec = sample[layer, head, :]  # shape [128]
        head_vectors.append(vec)
    head_vectors = np.stack(head_vectors)  # shape: [num_top_heads, 128]
    selected.append(head_vectors)

# Final output
selected = np.stack(selected)  # shape: [n, num_top_heads, 128]

flat_selected = act[:10000].reshape(selected.shape[0], -1)  
labels = labels[:10000]
pca = PCA(n_components=3)  # or 3 for 3D
X_reduced = pca.fit_transform(flat_selected)
print("done PCA ")
clf = SVC(kernel='linear', class_weight='balanced')  #SVC(C=3, gamma='auto', kernel='rbf')
clf.fit(X_reduced[:9000], labels[:9000])
train_acc = accuracy_score(labels[:9000], clf.predict(X_reduced[:9000]))
test_acc = accuracy_score(labels[9000:], clf.predict(X_reduced[9000:]))
print(clf.predict(X_reduced[-100:]), labels[-100:])
print(train_acc, test_acc)