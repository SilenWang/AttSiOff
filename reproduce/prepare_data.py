#!/usr/bin/env python3
"""
Convert ENsiRNA dataset to AttSiOff format using RNA-FM embeddings.
Usage:
  python prepare_data.py --data_dir <ENsiRNA_dataset_dir> [--output_dir ./data]
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import fm

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(repo_root, ".."))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    required=True,
    help="ENsiRNA dataset dir (containing train_*.csv, valid_*.csv)",
)
parser.add_argument(
    "--output_dir",
    default=os.path.join(repo_root, "data"),
    help="Output directory for converted data",
)
args = parser.parse_args()

DATA_DIR = args.data_dir
OUTPUT_DIR = args.output_dir
SIRNA_LEN = 21
MRNA_LEN = 59
EMBED_DIM = 640

print("Loading RNA-FM model...")
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    print("Warning: CUDA not available, using CPU")
model.to(device)
print(f"RNA-FM loaded on {device}")


def get_embedding(seq):
    _, _, tokens = batch_converter([("seq", seq.upper().replace("T", "U"))])
    with torch.no_grad():
        results = model(tokens.to(device), repr_layers=[12])
        return results["representations"][12][0][1:-1].cpu().numpy().astype(np.float32)


def extract_mrna_window(mrna_seq, position, anti_len=19, window_len=59):
    pad_len = (window_len - anti_len) // 2
    start = position - pad_len
    end = position + anti_len + pad_len
    if start < 0:
        left_pad = "." * (-start)
        window = left_pad + mrna_seq[:end]
    elif end > len(mrna_seq):
        right_pad = "." * (end - len(mrna_seq))
        window = mrna_seq[start:] + right_pad
    else:
        window = mrna_seq[start:end]
    return window.upper()


def process_fold(k):
    print(f"\n=== Processing fold {k} ===")
    train_csv = os.path.join(DATA_DIR, f"train_{k}.csv")
    valid_csv = os.path.join(DATA_DIR, f"valid_{k}.csv")
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    fold_dir = os.path.join(OUTPUT_DIR, f"fold_{k}")
    sirna_dir = os.path.join(fold_dir, "RNAFM_sirna")
    mrna_dir = os.path.join(fold_dir, "RNAFM_mrna")
    os.makedirs(sirna_dir, exist_ok=True)
    os.makedirs(mrna_dir, exist_ok=True)

    def convert_split(df, split_name):
        rows = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
            anti_seq = row["anti seq"].upper().replace("T", "U")
            anti_seq_21 = anti_seq + "AA"
            mrna_window = extract_mrna_window(row["mRNA_seq"], row["position"])
            rna_idx = f"{idx:04d}"

            np.save(
                os.path.join(sirna_dir, f"{rna_idx}.npy"), get_embedding(anti_seq_21)
            )
            np.save(
                os.path.join(mrna_dir, f"{rna_idx}.npy"), get_embedding(mrna_window)
            )

            rows.append(
                {
                    "Antisense": anti_seq_21,
                    "mrna": mrna_window,
                    "s-Biopredsi": 0.0,
                    "DSIR": 0.0,
                    "i-score": 0.0,
                    "inhibition": row["efficacy"],
                    "RNAFM_ind": idx,
                    "source_paper": split_name,
                }
            )
        return pd.DataFrame(rows)

    train_out = convert_split(train_df, "train")
    valid_out = convert_split(valid_df, "valid")
    combined = pd.concat([train_out, valid_out], ignore_index=True)
    csv_path = os.path.join(fold_dir, "normalized_sirna_with_mrna.csv")
    combined.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(combined)} rows)")
    return len(train_out), len(valid_out)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for k in range(1, 6):
        n_train, n_valid = process_fold(k)
        print(f"Fold {k}: {n_train} train, {n_valid} valid")
    print("\nDone!")
