#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Dataset
# -------------------------
class SeqDS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)  # (N,L,6)
        self.Y = Y.astype(np.float32)  # (N,2)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


def onehot_X_from_seqs(seqs: List[str], L: int) -> np.ndarray:
    base = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
    }
    X = np.zeros((len(seqs), L, 6), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper()
        oh4 = np.array([base.get(ch, [0, 0, 0, 0]) for ch in s], dtype=np.float32)
        if oh4.shape[0] < L:
            oh4 = np.vstack([oh4, np.zeros((L - oh4.shape[0], 4), np.float32)])
        else:
            oh4 = oh4[:L]
        X[i, :, :4] = oh4
    return X


def load_external_a549_csv(csv_path: Path, seq_col: str, L: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    External CSV format:
      Nn,cyt.score,nuc.score
    We build Y as [nuc, cyt] to match training output order [A549_nuc, A549_cyt, ...]
    """
    df = pd.read_csv(csv_path)
    need = [seq_col, "cyt.score", "nuc.score"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}. Found columns={df.columns.tolist()[:50]}")

    seqs = df[seq_col].astype(str).tolist()
    X = onehot_X_from_seqs(seqs, L=L)

    # IMPORTANT: align to model outputs: [nuc, cyt]
    Y = df[["nuc.score", "cyt.score"]].to_numpy(np.float32)
    return X, Y


# -------------------------
# Model
# -------------------------
class CNN1(nn.Module):
    def __init__(self, kernel_size, in_ch=6, cnn_out=256, mlp_hidden=128, out_dim=4, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=cnn_out, kernel_size=kernel_size, stride=1)
        self.act = nn.ReLU()
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.mlp = nn.Linear(cnn_out, mlp_hidden)
        self.head = nn.Linear(mlp_hidden, out_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)      # (B,L,6)->(B,6,L)
        x = self.act(self.conv(x))
        x = self.gmp(x).squeeze(-1) # (B,C)
        x = self.drop(x)
        x = self.act(self.mlp(x))
        return self.head(x)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    P, T = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yhat = model(xb).detach().cpu().numpy()
        P.append(yhat)
        T.append(yb.numpy())
    return np.concatenate(P, 0), np.concatenate(T, 0)


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.size == 0 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def eval_metrics_2d(pred2: np.ndarray, true2: np.ndarray) -> Dict[str, float]:
    # pred2/true2: (N,2) with order [nuc, cyt]
    out = {}
    names = ["nuc", "cyt"]
    maes, rs, r2s = [], [], []

    for j, nm in enumerate(names):
        mae = float(np.mean(np.abs(pred2[:, j] - true2[:, j])))
        r = pearson_r(pred2[:, j], true2[:, j])
        r2 = float(r * r) if np.isfinite(r) else np.nan
        out[f"mae_{nm}"] = mae
        out[f"r_{nm}"] = r
        out[f"r2_{nm}"] = r2
        maes.append(mae)
        rs.append(r)
        r2s.append(r2)

    out["mae_mean"] = float(np.mean(maes))
    out["r_mean"] = float(np.nanmean(rs))
    out["r2_mean"] = float(np.nanmean(r2s))
    return out


# -------------------------
# Checkpoint handling
# -------------------------
def parse_seed_ks(ckpt_path: Path) -> Tuple[Optional[int], Optional[int]]:
    seed = None
    ks = None
    m1 = re.search(r"seed(\d+)", str(ckpt_path))
    if m1:
        seed = int(m1.group(1))
    m2 = re.search(r"ks(\d+)\.pt$", ckpt_path.name)
    if m2:
        ks = int(m2.group(1))
    return seed, ks


def load_model_from_ckpt_first2(ckpt_path: Path, device: torch.device) -> nn.Module:
    """
    Load CNN1 checkpoint and keep only first 2 outputs:
      [A549_nuc, A549_cyt]
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    model = CNN1(
        kernel_size=int(ckpt["kernel_size"]),
        in_ch=int(ckpt.get("in_ch", 6)),
        cnn_out=int(ckpt["cnn_out"]),
        mlp_hidden=int(ckpt["mlp_hidden"]),
        out_dim=2,
        dropout=float(ckpt.get("dropout", 0.0)),
    )

    state = ckpt["state_dict"]
    # truncate head
    if state["head.weight"].shape[0] < 2:
        raise ValueError(f"{ckpt_path}: checkpoint head out_dim={state['head.weight'].shape[0]} < 2")
    state = dict(state)
    state["head.weight"] = state["head.weight"][:2].clone()
    state["head.bias"] = state["head.bias"][:2].clone()

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=str, default="./cnn1_kernel_sweep_out_seeds/models",
                    help="Directory containing seed*/ks*.pt")
    ap.add_argument("--test_csv", type=str, default="./train_data_260128/3pL6-A549-T1.csv")
    ap.add_argument("--seq_col", type=str, default="Nn")
    ap.add_argument("--seq_len", type=int, default=46)

    ap.add_argument("--eval_bs", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)

    ap.add_argument("--out_csv", type=str, default="./cnn1_kernel_sweep_out_seeds/external_test_3pL6-A549-T1.metrics.csv")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    models_dir = Path(args.models_dir)
    test_csv = Path(args.test_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # load external test: Y=[nuc,cyt]
    X, Y = load_external_a549_csv(test_csv, seq_col=args.seq_col, L=args.seq_len)
    loader = DataLoader(SeqDS(X, Y), batch_size=args.eval_bs, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    ckpts = sorted(models_dir.glob("seed*/ks*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {models_dir} (expected seed*/ks*.pt)")

    rows = []
    for ckpt_path in ckpts:
        seed, ks = parse_seed_ks(ckpt_path)

        model = load_model_from_ckpt_first2(ckpt_path, device=device)
        pred2, true2 = predict(model, loader, device=device)

        m = eval_metrics_2d(pred2, true2)
        row = {
            "ckpt_path": str(ckpt_path),
            "seed": seed,
            "kernel_size": ks,
            "test_csv": str(test_csv),
            "N": int(Y.shape[0]),
        }
        row.update(m)
        rows.append(row)

        print(f"[OK] seed={seed} ks={ks} | mae_mean={row['mae_mean']:.4f} r2_mean={row['r2_mean']:.4f}")

    df = pd.DataFrame(rows).sort_values(["kernel_size", "seed"])
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
