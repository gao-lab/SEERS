#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# -------------------------
# Tokenization (match your Keras vocab)
# vocab = ['pad','N','A','T','C','G']
# -------------------------
VOCAB = ["pad", "N", "A", "T", "C", "G"]
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}


def kmerize(seq: str, k: int) -> List[str]:
    seq = str(seq).upper()
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def vectorize_dna_seq(dna_seq: str) -> List[int]:
    # Keras 版：直接按 char2idx[char]，这里对未知字符按 'N'
    dna_seq = str(dna_seq).upper()
    out = []
    for ch in dna_seq:
        out.append(CHAR2IDX.get(ch, CHAR2IDX["N"]))
    return out


def pad_to_len(tokens: List[int], L: int, pad_idx: int = 0) -> List[int]:
    if len(tokens) >= L:
        return tokens[:L]
    return tokens + [pad_idx] * (L - len(tokens))


def prepare_x_token_ids(dna_list: List[str], x_len: int) -> np.ndarray:
    # output: (N, x_len) int64
    xs = []
    for s in dna_list:
        tok = vectorize_dna_seq(s)
        tok = pad_to_len(tok, x_len, pad_idx=CHAR2IDX["pad"])
        xs.append(tok)
    return np.asarray(xs, dtype=np.int64)


# -------------------------
# Model (adjust to match your checkpoint)
# -------------------------
class SEERSLSTM2Head(nn.Module):
    """
    A minimal LSTM model for token ids (B, L) -> (B, 2).
    Default tries to mirror "LSTM64x32" style.

    IMPORTANT: You must match this architecture to your saved .pth checkpoint.
    """

    def __init__(
        self,
        vocab_size: int = 6,
        embed_dim: int = 5,
        lstm_hidden1: int = 64,
        lstm_hidden2: int = 32,
        dropout: float = 0.5,
        fc_hidden: int = 64,
        out_dim: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstm_hidden1, hidden_size=lstm_hidden2, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden2, fc_hidden)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, out_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, L) long
        x = self.embedding(token_ids)              # (B, L, E)
        x, _ = self.lstm1(x)                       # (B, L, H1)
        x, _ = self.lstm2(x)                       # (B, L, H2)
        x = x[:, -1, :]                            # take last step (B, H2)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)                         # (B, out_dim)


# -------------------------
# Dataset for batched prediction
# -------------------------
class TokenDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx])


@torch.no_grad()
def batched_predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
    num_workers: int = 0,
) -> np.ndarray:
    ds = TokenDataset(X)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model.eval()
    out = []
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        y = model(xb).detach().cpu().numpy()
        out.append(y)
    return np.concatenate(out, axis=0)


def build_window_matrices_for_snps(
    ref_list: List[str],
    alt_list: List[str],
    kmer: int,
    x_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build big ref/alt window matrices for ALL SNPs for efficient prediction.

    Return:
      X_ref: (S*W, x_len)
      X_alt: (S*W, x_len)
      win_counts: (S,) number of windows per SNP (should be constant W=89-kmer+1 if all are length 89)
    """
    if len(ref_list) != len(alt_list):
        raise ValueError("ref_list and alt_list must have same length")

    Xref_all, Xalt_all = [], []
    win_counts = np.zeros(len(ref_list), dtype=np.int64)

    for i, (ref, alt) in enumerate(zip(ref_list, alt_list)):
        ref = str(ref).upper()
        alt = str(alt).upper()

        if len(ref) < kmer or len(alt) < kmer:
            # keep empty; caller will handle NaN
            win_counts[i] = 0
            continue

        ref_ws = kmerize(ref, kmer)
        alt_ws = kmerize(alt, kmer)
        if len(ref_ws) != len(alt_ws):
            raise ValueError(f"Window count mismatch at row {i}: ref={len(ref_ws)} alt={len(alt_ws)}")

        win_counts[i] = len(ref_ws)
        Xref_all.append(prepare_x_token_ids(ref_ws, x_len))
        Xalt_all.append(prepare_x_token_ids(alt_ws, x_len))

    if len(Xref_all) == 0:
        return np.zeros((0, x_len), np.int64), np.zeros((0, x_len), np.int64), win_counts

    X_ref = np.concatenate(Xref_all, axis=0)
    X_alt = np.concatenate(Xalt_all, axis=0)
    return X_ref, X_alt, win_counts


def compute_median_delta_per_snp(
    pred_ref: np.ndarray,
    pred_alt: np.ndarray,
    win_counts: np.ndarray,
    cyt_idx: int = 1,
) -> np.ndarray:
    """
    pred_ref/pred_alt are concatenated window predictions: shape (sum_W, out_dim)
    win_counts gives each SNP's W.
    Return: (S,) median delta on cyt_idx.
    """
    S = win_counts.shape[0]
    out = np.full(S, np.nan, dtype=np.float64)

    cur = 0
    for i in range(S):
        w = int(win_counts[i])
        if w <= 0:
            continue
        a = pred_alt[cur : cur + w, cyt_idx]
        r = pred_ref[cur : cur + w, cyt_idx]
        d = a - r
        out[i] = float(np.median(d))
        cur += w
    return out


# -------------------------
# Random SNP simulation
# -------------------------
def generate_random_dna(length: int = 89) -> str:
    return "".join(random.choice("ATGC") for _ in range(length))


def mutate_dna(dna: str, mutation_position: int) -> str:
    dna = str(dna).upper()
    original = dna[mutation_position]
    bases = "ATGC"
    mutated = random.choice([b for b in bases if b != original])
    return dna[:mutation_position] + mutated + dna[mutation_position + 1 :]


def main():
    ap = argparse.ArgumentParser()

    # model
    ap.add_argument("--model_path", type=str, required=True, help="PyTorch checkpoint (.pth). state_dict preferred.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=0)

    # clinvar
    ap.add_argument("--clinvar_tsv", type=str, required=True, help="Input TSV with ref/alt columns (e.g. seq89 / seq89.mutated).")
    ap.add_argument("--ref_col", type=str, default="seq89")
    ap.add_argument("--alt_col", type=str, default="seq89.mutated")
    ap.add_argument("--out_tsv", type=str, default="ClinVar_3UTR_SNPs_effects.pytorch.tsv")

    # windowing
    ap.add_argument("--kmer", type=int, default=45)
    ap.add_argument("--x_len", type=int, default=46, help="Pad length. Keep 46 to mimic original Keras script.")
    ap.add_argument("--cyt_idx", type=int, default=1, help="Which output dim corresponds to Cyt. Keras uses [:,1].")

    # model arch params (must match checkpoint)
    ap.add_argument("--embed_dim", type=int, default=5)
    ap.add_argument("--lstm_hidden1", type=int, default=64)
    ap.add_argument("--lstm_hidden2", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--fc_hidden", type=int, default=64)
    ap.add_argument("--out_dim", type=int, default=2)

    # random simulation
    ap.add_argument("--do_random", action="store_true", help="Also run random SNP simulation + histogram.")
    ap.add_argument("--n_random", type=int, default=10000)
    ap.add_argument("--mutation_pos", type=int, default=44, help="0-based index of SNP in 89nt (center=44).")
    ap.add_argument("--random_hist_png", type=str, default="random_snp_effects.pytorch.png")
    ap.add_argument("--hist_bins", type=int, default=2048)
    ap.add_argument("--hist_xlim", type=str, default="-0.2,0.2")

    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"--model_path not found: {model_path}")

    # build model
    model = SEERSLSTM2Head(
        vocab_size=len(VOCAB),
        embed_dim=args.embed_dim,
        lstm_hidden1=args.lstm_hidden1,
        lstm_hidden2=args.lstm_hidden2,
        dropout=args.dropout,
        fc_hidden=args.fc_hidden,
        out_dim=args.out_dim,
    )

    # load checkpoint
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # read clinvar
    df = pd.read_csv(args.clinvar_tsv, sep="\t")
    if args.ref_col not in df.columns or args.alt_col not in df.columns:
        raise ValueError(f"Missing columns. Need ref_col={args.ref_col}, alt_col={args.alt_col}. "
                         f"Found={df.columns.tolist()[:60]}")

    ref_list = df[args.ref_col].astype(str).tolist()
    alt_list = df[args.alt_col].astype(str).tolist()

    # build window matrices
    X_ref, X_alt, win_counts = build_window_matrices_for_snps(
        ref_list=ref_list,
        alt_list=alt_list,
        kmer=args.kmer,
        x_len=args.x_len,
    )

    # predict
    pred_ref = batched_predict(model, X_ref, device=device, batch_size=args.batch_size, num_workers=args.workers)
    pred_alt = batched_predict(model, X_alt, device=device, batch_size=args.batch_size, num_workers=args.workers)

    # median delta
    delta = compute_median_delta_per_snp(
        pred_ref=pred_ref,
        pred_alt=pred_alt,
        win_counts=win_counts,
        cyt_idx=args.cyt_idx,
    )

    df["delta.log2.cyt"] = delta
    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[OK] Saved ClinVar effects: {out_tsv} (N={len(df)})")

    # random simulation
    if args.do_random:
        lo, hi = [float(x) for x in args.hist_xlim.split(",")]
        results = []
        for _ in range(int(args.n_random)):
            ref = generate_random_dna(89)
            alt = mutate_dna(ref, mutation_position=int(args.mutation_pos))

            ref_ws = kmerize(ref, args.kmer)
            alt_ws = kmerize(alt, args.kmer)
            Xr = prepare_x_token_ids(ref_ws, args.x_len)
            Xa = prepare_x_token_ids(alt_ws, args.x_len)

            pr = batched_predict(model, Xr, device=device, batch_size=args.batch_size, num_workers=0)
            pa = batched_predict(model, Xa, device=device, batch_size=args.batch_size, num_workers=0)

            d = pa[:, args.cyt_idx] - pr[:, args.cyt_idx]
            results.append(float(np.median(d)))

        results = np.asarray(results, dtype=np.float64)
        png = Path(args.random_hist_png)
        png.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 4))
        plt.hist(results, bins=int(args.hist_bins), edgecolor="black", alpha=0.7)
        plt.xlim(lo, hi)
        plt.title("Random SNP effects (PyTorch)")
        plt.xlabel("Delta log2(Cyt/DNA)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(png, dpi=300)
        plt.close()
        print(f"[OK] Saved random SNP histogram: {png}")


if __name__ == "__main__":
    main()
