#!/usr/bin/env python3
"""
Precompute and save sentence embeddings (prompt encodings) for texts in a dataset.
Similar to build_topk_neighbors.py but specifically for prompt encodings used in special directions.
"""

import os
import argparse
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def embed_texts_sentence(texts, st_model="sentence-transformers/all-MiniLM-L6-v2", batch=512):
    """Encode texts in batches using sentence transformer."""
    model = SentenceTransformer(st_model)
    vecs = []
    for i in tqdm(range(0, len(texts), batch), desc=f"embed({st_model})"):
        chunk = texts[i:i+batch]
        arr = model.encode(chunk, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        vecs.append(arr.astype("float32"))
    X = np.vstack(vecs).astype("float32")
    # Normalize to unit vectors
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (norms + 1e-8)
    return X.astype("float32")


def main():
    parser = argparse.ArgumentParser(description="Precompute prompt encodings for texts")
    parser.add_argument("--dataset_name", type=str, required=True, 
                        help="Dataset name (e.g., toxigen_vicuna, hate_vicuna)")
    parser.add_argument("--model_name", type=str, default="llama3_8B",
                        help="Model name for output file naming")
    parser.add_argument("--st_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model to use")
    parser.add_argument("--batch_embed", type=int, default=512,
                        help="Batch size for encoding")
    parser.add_argument("--out_dir", type=str, 
                        default="/work/hdd/bcxt/yian3/toxic/local_store_toxigen",
                        help="Output directory for saved encodings")
    
    args = parser.parse_args()
    
    # Load dataframe based on dataset name
    print(f"[load] dataset={args.dataset_name}")
    if args.dataset_name == "toxigen_vicuna":
        toxigen_path = f'/work/hdd/bcxt/yian3/toxic/features/{args.dataset_name}_texts.json'
        with open(toxigen_path) as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
    elif args.dataset_name == "hate_vicuna":
        hate_path = f'/work/hdd/bcxt/yian3/toxic/features/{args.dataset_name}_texts.json'
        with open(hate_path) as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
    else:
        # Try to load from CSV
        csv_path = f'./TruthfulQA/{args.dataset_name}.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    # Extract texts
    if 'text' not in df.columns:
        raise ValueError(f"Dataframe must have 'text' column. Available columns: {df.columns.tolist()}")
    
    texts = df['text'].tolist()
    print(f"[load] {len(texts)} texts")
    
    # Encode texts
    print(f"[embed] encoding texts with {args.st_model}")
    prompt_encodings = embed_texts_sentence(texts, st_model=args.st_model, batch=args.batch_embed)
    
    # Save encodings
    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, f"texts.npy")
    np.save(output_path, prompt_encodings)
    print(f"[save] {output_path}")
    print(f"[save] texts.npy {prompt_encodings.shape} float32 (unit-norm rows)")
    print(f"[done] Saved {len(texts)} prompt encodings")


if __name__ == "__main__":
    main()

"""
python build_prompt_encodings.py \
  --dataset_name toxigen_vicuna \
  --out_dir /work/hdd/bcxt/yian3/toxic/local_store_toxigen  \
  --st_model all-MiniLM-L6-v2 \
  --batch_embed 512
"""