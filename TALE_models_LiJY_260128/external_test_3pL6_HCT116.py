#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
from pathlib import Path
from typing import List, Tuple, Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from scipy.stats import linregress


def scatter_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_png: Path,
    figsize=(5, 5),
    s=6,
    alpha=0.5,
):
    """
    Scatter: Observed (x) vs Predicted (y), with y=x line and Pearson R^2.
    """
    if y_true.size == 0:
        return

    slope, intercept, r_value, _, _ = linregress(y_true, y_pred)
    r2 = r_value ** 2

    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, s=s, alpha=alpha)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"{title}\nPearson R² = {r2:.3f}")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


# -------------------------
# Dataset
# -------------------------
class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)  # (N,L,6)
        self.Y = Y.astype(np.float32)  # (N,2)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


def onehot6_from_seqs(seqs: List[str], L: int) -> np.ndarray:
    base_to_vec = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
    }
    X = np.zeros((len(seqs), L, 6), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper()
        oh4 = np.array([base_to_vec.get(ch, [0, 0, 0, 0]) for ch in s], dtype=np.float32)
        if oh4.shape[0] < L:
            oh4 = np.vstack([oh4, np.zeros((L - oh4.shape[0], 4), dtype=np.float32)])
        else:
            oh4 = oh4[:L]
        X[i, :, :4] = oh4
    return X


def load_external_hct116_csv(csv_path: Path, seq_col: str = "Nn", L: int = 46) -> Tuple[np.ndarray, np.ndarray]:
    """
    External CSV must contain columns:
      seq_col, cyt.score, nuc.score
    Build Y as [nuc, cyt].
    """
    df = pd.read_csv(csv_path)
    need = [seq_col, "cyt.score", "nuc.score"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}. Found columns={df.columns.tolist()[:60]}")

    seqs = df[seq_col].astype(str).tolist()
    X = onehot6_from_seqs(seqs, L=L)
    Y = df[["nuc.score", "cyt.score"]].to_numpy(np.float32)
    return X, Y


# -------------------------
# Model (LSTM, same as training)
# -------------------------
class SEERSModel(nn.Module):
    def __init__(self, model_type='lstm', vocab_size=6, embed_dim=5, seq_length=150, output_dim=2):
        super(SEERSModel, self).__init__()
        self.model_type = model_type
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        if model_type == 'lstm':
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
            self.lstm1 = nn.LSTM(input_size=embed_dim, hidden_size=128, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
            self.dropout1 = nn.Dropout(0.5)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * seq_length, 128)
            self.dropout2 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, output_dim)
        else:
            raise ValueError("This eval script is for LSTM models only.")

    def _to_token_ids_from_onehot6(self, x6):
        bases = x6[:, :, :4]
        token_ids = bases.argmax(dim=-1)
        is_N = bases.sum(dim=-1) == 0
        n_idx = min(self.vocab_size - 1, 4)
        token_ids = token_ids.masked_fill(is_N, n_idx)
        return token_ids.long()

    def forward(self, x):
        if x.dim() == 3 and x.size(-1) == 6:
            token_ids = self._to_token_ids_from_onehot6(x)
        elif x.dim() == 2 and x.dtype in (torch.long, torch.int64):
            token_ids = x
        else:
            raise ValueError("Expected (B,L,6) onehot6 or (B,L) token ids.")

        x = self.embedding(token_ids)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


# -------------------------
# Metrics
# -------------------------
def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.size == 0 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def eval_metrics_2d(pred2: np.ndarray, true2: np.ndarray) -> Dict[str, float]:
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


def parse_prefix_list(s: str) -> List[int]:
    """
    Accept:
      - "[1000,2000,5000]"
      - "1000,2000,5000"
      - "1000 2000 5000"
    Return sorted unique positive ints, preserving given order where possible.
    """
    s = (s or "").strip()
    if not s:
        return []
    # try python literal first
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                out = [int(x) for x in v]
                # keep order, drop non-positive, de-dup
                seen = set()
                res = []
                for k in out:
                    if k <= 0:
                        continue
                    if k not in seen:
                        seen.add(k)
                        res.append(k)
                return res
        except Exception:
            pass

    # fallback split by comma/space
    parts = [p for p in s.replace(",", " ").split() if p]
    out = []
    seen = set()
    for p in parts:
        k = int(p)
        if k <= 0:
            continue
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def eval_prefix_r2(pred2: np.ndarray, true2: np.ndarray, prefix_list: Sequence[int]) -> Dict[str, float]:
    """
    For each k in prefix_list, compute r2 for nuc/cyt and mean.
    Keys:
      prefix{k}_r2_nuc, prefix{k}_r2_cyt, prefix{k}_r2_mean
    """
    out = {}
    n = pred2.shape[0]
    for k in prefix_list:
        kk = min(int(k), n)
        if kk <= 1:
            out[f"prefix{k}_r2_nuc"] = np.nan
            out[f"prefix{k}_r2_cyt"] = np.nan
            out[f"prefix{k}_r2_mean"] = np.nan
            continue
        m = eval_metrics_2d(pred2[:kk], true2[:kk])
        out[f"prefix{k}_r2_nuc"] = float(m["r2_nuc"])
        out[f"prefix{k}_r2_cyt"] = float(m["r2_cyt"])
        out[f"prefix{k}_r2_mean"] = float(m["r2_mean"])
    return out


@torch.no_grad()
def predict_hct116_2d(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    model output: (N,4) = [A549_nuc, A549_cyt, HCT116_nuc, HCT116_cyt]
    return pred2: (N,2) = [HCT116_nuc, HCT116_cyt]
    """
    model.eval()
    P2, T2 = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yhat4 = model(xb).detach().cpu().numpy()
        P2.append(yhat4[:, 2:4])   # HCT116
        T2.append(yb.numpy())
    return np.concatenate(P2, 0), np.concatenate(T2, 0)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="Path to ONE LSTM final_model.pth (state_dict)")
    ap.add_argument("--test_csv", type=str, required=True, help="External HCT116 CSV with Nn,cyt.score,nuc.score")
    ap.add_argument("--seq_col", type=str, default="Nn")
    ap.add_argument("--seq_len", type=int, default=46)

    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval_bs", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=4)

    ap.add_argument(
        "--prefix_list",
        type=str,
        default="[1000,2000,5000]",
        help='Prefix sizes list. Example: "[1000,2000,5000,10000,20000]" or "1000,2000,5000,10000,20000"'
    )
    ap.add_argument("--out_csv", type=str, default="./external_test_HCT116.single_final.lstm_metrics.csv")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"--model_path not found: {model_path}")

    test_csv = Path(args.test_csv)
    if not test_csv.exists():
        raise FileNotFoundError(f"--test_csv not found: {test_csv}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    prefix_list = parse_prefix_list(args.prefix_list)
    if not prefix_list:
        prefix_list = [1000, 2000, 5000]  # safe fallback

    # data
    X, Y = load_external_hct116_csv(test_csv, seq_col=args.seq_col, L=args.seq_len)
    loader = DataLoader(
        SequenceDataset(X, Y),
        batch_size=args.eval_bs,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # model (must match training arch)
    model = SEERSModel(model_type="lstm", seq_length=args.seq_len, output_dim=4)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    pred2, true2 = predict_hct116_2d(model, loader, device=device)
    # -------------------------
    # Scatter plots for each prefix
    # -------------------------
    fig_root = Path("./figure/test_3pL4-HCT116-T1")
    fig_root.mkdir(parents=True, exist_ok=True)

    for k in prefix_list:
        kk = min(int(k), pred2.shape[0])
        if kk <= 1:
            continue

        # nuc
        scatter_pred_vs_true(
            y_true=true2[:kk, 0],
            y_pred=pred2[:kk, 0],
            title=f"HCT116 nuc (prefix {kk})",
            out_png=fig_root / f"prefix{kk}_nuc.png",
        )

        # cyt
        scatter_pred_vs_true(
            y_true=true2[:kk, 1],
            y_pred=pred2[:kk, 1],
            title=f"HCT116 cyt (prefix {kk})",
            out_png=fig_root / f"prefix{kk}_cyt.png",
        )

    m = eval_metrics_2d(pred2, true2)
    pm = eval_prefix_r2(pred2, true2, prefix_list=prefix_list)

    row = {
        "model_path": str(model_path),
        "test_csv": str(test_csv),
        "N": int(true2.shape[0]),
        "prefix_list": ",".join(str(x) for x in prefix_list),
    }
    row.update(m)
    row.update(pm)

    print(f"[OK] model={model_path} | mae_mean={row['mae_mean']:.4f} r2_mean={row['r2_mean']:.4f}")
    for k in prefix_list:
        print(f"  prefix{k}: r2_nuc={row[f'prefix{k}_r2_nuc']:.4f} "
              f"r2_cyt={row[f'prefix{k}_r2_cyt']:.4f} "
              f"r2_mean={row[f'prefix{k}_r2_mean']:.4f}")

    pd.DataFrame([row]).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
