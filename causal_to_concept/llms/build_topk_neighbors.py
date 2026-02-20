#!/usr/bin/env python3
"""
build_keys_and_neighbors.py

Given a dataset *name*, resolve its data path + schema, then:
  1) compute sentence-embedding context keys for the non-toxic `text_field`
  2) (optional) also compute keys for `toxic_field` (for contrastive weighting)
  3) precompute top-K nearest neighbors over the non-toxic keys
Outputs in --out_dir:
  - keys.npy              : float32 [N, d] unit-norm
  - (optional) keys_toxic.npy : float32 [N, d] unit-norm
  - neighbors_topK.npy    : int32 [N, K]
  - sims_topK.npy         : float16 [N, K]
  - ids.json, meta.json

Usage:
  python build_keys_and_neighbors.py --dataset myset --out_dir artifacts_ctx --K 512
"""

import os, json, argparse, numpy as np
import time
import torch
from tqdm import tqdm, trange
from typing import Dict, Any
import faiss
import pandas as pd

# ---------------------------
# 1) Register datasets here
# ---------------------------
def load_data_like_validate_2fold(dataset_name):
    """Load data the same way as validate_2fold_toxic.py"""
    if dataset_name == "toxigen_vicuna":
        toxigen_path = f'/work/hdd/bcxt/yian3/toxic/features/{dataset_name}_texts.json'
        with open(toxigen_path) as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
    elif dataset_name == "hate_vicuna":
        hate_path = f'/work/hdd/bcxt/yian3/toxic/features/{dataset_name}_texts.json'
        with open(hate_path) as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
    else:
        # Fallback to CSV loading
        df = pd.read_csv(f'./TruthfulQA/{dataset_name}.csv')
    
    print(f"Loaded {len(df)} examples from {dataset_name}")
    print(f"Columns: {df.columns.tolist()}")
    return df

# ---------------------------
# 2) Extract texts
# ---------------------------
def load_all_data(dataset_name):
    """Load all data pairs (toxic vs non-toxic) from the dataset."""
    # Load the full dataset
    df = load_data_like_validate_2fold(dataset_name)
    
    # Extract texts from the dataframe
    if 'toxic_text' in df.columns and 'non_toxic_text' in df.columns:
        toxic_texts = df['toxic_text'].dropna().tolist()
        nontoxic_texts = df['non_toxic_text'].dropna().tolist()
        texts = df['text'].dropna().tolist()
        print(f"Using toxic_text and non_toxic_text columns")
    else:
        raise ValueError(f"Could not find 'toxic_text'/'non_toxic_text' columns in {df.columns.tolist()}")
    
    # Ensure same length
    min_len = min(len(toxic_texts), len(nontoxic_texts))
    texts = texts[:min_len]
    toxic_texts = toxic_texts[:min_len]
    nontoxic_texts = nontoxic_texts[:min_len]
    
    print(f"Final dataset size: {len(toxic_texts)} pairs")
    
    return texts, toxic_texts, nontoxic_texts

def build_faiss_index(keys: np.ndarray, index_type: str = "flat", device: str = "cpu",
                      nlist: int = 4096, nprobe: int = 16):
    d = keys.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(d)  # exact, inner product
    elif index_type == "ivf":
        # IVF with Flat residuals (no PQ) for high recall
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    else:
        raise ValueError("index_type must be 'flat' or 'ivf'.")

    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    if index_type == "ivf":
        # IVF needs training
        index.train(keys)
        # set probes
        index.nprobe = nprobe

    index.add(keys)
    return index

def faiss_topk_self(keys: np.ndarray, index, K: int) -> (np.ndarray, np.ndarray):
    """
    Query each row of keys against the same index to get top-K neighbors (exclude self).
    Returns:
      neighbors: int32 [N, K]
      sims: float16 [N, K]
    """
    N, d = keys.shape
    # search for K+1 then drop self (first hit is typically self)
    D, I = index.search(keys, K + 1)  # D: [N, K+1], I: [N, K+1]
    neighbors = np.empty((N, K), dtype=np.int32)
    sims = np.empty((N, K), dtype=np.float16)
    for i in range(N):
        ids = I[i]
        sims_i = D[i]
        # drop exact self idx if present
        if ids[0] == i:
            ids = ids[1:K+1]
            sims_row = sims_i[1:K+1]
        else:
            ids = ids[:K]
            sims_row = sims_i[:K]
        neighbors[i] = ids.astype(np.int32)
        sims[i] = sims_row.astype(np.float16)
    return neighbors, sims

# ---------------------------
# 3) Embedding + neighbors
# ---------------------------
def embed_texts_sentence(texts, st_model="sentence-transformers/all-MiniLM-L6-v2", batch=512):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(st_model)
    vecs = []
    for i in tqdm(range(0, len(texts), batch), desc=f"embed({st_model})"):
        chunk = texts[i:i+batch]
        arr = model.encode(chunk, convert_to_numpy=True, normalize_embeddings=False)
        vecs.append(arr.astype("float32"))
    X = np.vstack(vecs).astype("float32")
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X /= norms  # unit-norm rows -> cosine = inner product
    return X

def precompute_topk_neighbors(keys_np, K=512, batch=2048, device=None):
    # device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # N, d = keys_np.shape
    # # ensure normalized
    # keys = keys_np / (np.linalg.norm(keys_np, axis=1, keepdims=True) + 1e-8)
    # keys_t = torch.from_numpy(keys).to(device)
    # neighbors = np.empty((N, K), dtype=np.int32)
    # sims = np.empty((N, K), dtype=np.float16)
    # neg_inf = -1e9
    # with torch.no_grad():
    #     for start in trange(0, N, batch, desc=f"topK(K={K}, batch={batch}, dev={device})"):
    #         end = min(start + batch, N)
    #         Q = keys_t[start:end]               # [B, d]
    #         S = Q @ keys_t.T                    # [B, N] cosine (inner product)
    #         ar = torch.arange(start, end, device=device)
    #         S[torch.arange(end-start, device=device), ar] = neg_inf  # exclude self
    #         topv, topi = torch.topk(S, k=K, dim=1, largest=True, sorted=True)
    #         neighbors[start:end] = topi.cpu().numpy().astype(np.int32)
    #         sims[start:end] = topv.cpu().numpy().astype(np.float16)
    # Build FAISS index
    index = build_faiss_index(keys_np, index_type="flat", device=device)
    
    # Query all vectors against the index
    neighbors, sims = faiss_topk_self(keys_np, index, K)
    
    return neighbors, sims

# ---------------------------
# 4) Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset name key (registered in this script)")
    ap.add_argument("--root", default=None, help="Optional root to prepend to dataset path")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--st_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_embed", type=int, default=512)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--batch_topk", type=int, default=2048)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--timing_knn", action="store_true", help="Print wall-clock time for top-K neighbor precompute (start/end timing)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # Load all data
    print("Loading all data...")
    texts, toxic_texts, nontoxic_texts = load_all_data(args.dataset)

    
    keys = embed_texts_sentence(texts, st_model=args.st_model, batch=args.batch_embed)
    np.save(os.path.join(args.out_dir, "keys.npy"), keys)
    print(f"[save] keys.npy {keys.shape} float32 (unit-norm rows)")

    # toxic keys (for contrastive weighting later)
    print(f"[embed] toxic_field='toxic_texts'")
    keys_tox = embed_texts_sentence(toxic_texts, st_model=args.st_model, batch=args.batch_embed)
    np.save(os.path.join(args.out_dir, "keys_toxic.npy"), keys_tox)
    print(f"[save] keys_toxic.npy {keys_tox.shape} float32 (unit-norm rows)")

    # Save ids + meta
    with open(os.path.join(args.out_dir, "ids.json"), "w") as f:
        json.dump([{"row": i, "id": i} for i in range(len(texts))], f)
    
    meta = dict(
        dataset=args.dataset,
        path=args.out_dir,
        format="json",
        text_field="texts",
        toxic_field="toxic_texts",
        key_type="sentence_embed",
        st_model=args.st_model,
        N=len(texts),
        dim=int(keys.shape[1]),
        normalized=True
    )
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[save] meta.json & ids.json")

    # Precompute top-K neighbors on non-toxic keys
    print(f"[topK] computing K={args.K} neighbors")
    if args.timing_knn and args.device == "cuda" and torch.cuda.is_available():
        # Best-effort sync for more meaningful wall-clock timing when CUDA is used.
        torch.cuda.synchronize()
    t0 = time.perf_counter() if args.timing_knn else None
    neighbors, sims = precompute_topk_neighbors(keys, K=args.K, batch=args.batch_topk, device=args.device)
    if args.timing_knn and args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter() if args.timing_knn else None
    np.save(os.path.join(args.out_dir, f"neighbors_top{args.K}.npy"), neighbors)
    np.save(os.path.join(args.out_dir, f"sims_top{args.K}.npy"), sims)
    print(f"[save] neighbors_top{args.K}.npy {neighbors.shape} int32")
    print(f"[save] sims_top{args.K}.npy {sims.shape} float16")
    if args.timing_knn:
        print(f"[timing] knn_precompute_sec: {t1 - t0:.6f} (N={keys.shape[0]}, K={args.K}, device={args.device})")

if __name__ == "__main__":
    main()


"""
python build_topk_neighbors.py \
  --dataset toxigen_vicuna \
  --out_dir /work/hdd/bcxt/yian3/toxic/local_store_toxigen  \
  --K 256 \
  --device cpu \
  --timing_knn

"""