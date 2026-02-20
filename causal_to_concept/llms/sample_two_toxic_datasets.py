import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import List

PREFERRED_COLUMNS = [
    "text",
    "non_toxic_text",
    "toxic_text",
    "non_toxic_tox_score",
    "toxic_tox_score",
    "non_toxic_fluency_score",
    "toxic_fluency_score",
]

def read_nonempty_lines(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def sample_lines(lines: List[str], sample_size: int, rng: random.Random) -> List[str]:
    if not lines:
        return []
    k = min(sample_size, len(lines))
    return rng.sample(lines, k)


def write_lines(file_path: str, lines: List[str]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def write_csv(file_path: str, lines: List[str]) -> None:
    parsed_rows = []
    for line in lines:
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                parsed_rows.append(row)
            else:
                parsed_rows.append({"text": line})
        except json.JSONDecodeError:
            parsed_rows.append({"text": line})

    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREFERRED_COLUMNS)
        writer.writeheader()
        for row in parsed_rows:
            writer.writerow({k: row.get(k, "") for k in PREFERRED_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read two line-based datasets (e.g., JSONL) and randomly sample lines."
        )
    )
    parser.add_argument("--dataset1_path", type=str, required=True, help="Path to dataset 1")
    parser.add_argument("--dataset2_path", type=str, required=True, help="Path to dataset 2")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="Number of lines to sample from each dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./randomsample",
        help="Directory to save sampled files",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="csv",
        choices=["jsonl", "csv"],
        help="Output file format for sampled datasets",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    ds1_lines = read_nonempty_lines(args.dataset1_path)
    ds2_lines = read_nonempty_lines(args.dataset2_path)

    ds1_sample = sample_lines(ds1_lines, args.sample_size, rng)
    ds2_sample = sample_lines(ds2_lines, args.sample_size, rng)

    ds1_name = Path(args.dataset1_path).stem
    ds2_name = Path(args.dataset2_path).stem

    ext = "csv" if args.output_format == "csv" else "jsonl"
    ds1_out = os.path.join(args.output_dir, f"{ds1_name}_sampled_{len(ds1_sample)}.{ext}")
    ds2_out = os.path.join(args.output_dir, f"{ds2_name}_sampled_{len(ds2_sample)}.{ext}")

    if args.output_format == "csv":
        write_csv(ds1_out, ds1_sample)
        write_csv(ds2_out, ds2_sample)
    else:
        write_lines(ds1_out, ds1_sample)
        write_lines(ds2_out, ds2_sample)

    combined = ds1_sample + ds2_sample
    rng.shuffle(combined)
    combined_out = os.path.join(
        args.output_dir, f"combined_{ds1_name}_{ds2_name}_sampled_{len(combined)}.{ext}"
    )
    if args.output_format == "csv":
        write_csv(combined_out, combined)
    else:
        write_lines(combined_out, combined)

    print(f"Dataset 1: {len(ds1_lines)} lines -> sampled {len(ds1_sample)}")
    print(f"Dataset 2: {len(ds2_lines)} lines -> sampled {len(ds2_sample)}")
    print(f"Saved: {ds1_out}")
    print(f"Saved: {ds2_out}")
    print(f"Saved: {combined_out}")


if __name__ == "__main__":
    main()

"""
python sample_two_toxic_datasets.py \
    --dataset1_path /work/hdd/bcxt/yian3/toxic/features/toxigen_vicuna_texts.json \
    --dataset2_path /work/hdd/bcxt/yian3/toxic/features/hate_vicuna_texts.json \
    --sample_size 25 --seed 42 --output_format csv
"""
