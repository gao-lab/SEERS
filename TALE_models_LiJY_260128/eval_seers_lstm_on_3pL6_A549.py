#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Dataset (same style as your training)
# -------------------------
class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)  # (N,L,6)
        self.Y = Y.astype(np.float32)  # (N,2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)


def onehot6_from_seqs(seqs: List[str], L: int) -> np.ndarray:
    base_to_vec = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0],
    }
    X = np.zeros((len(seqs), L, 6), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper()
        one_hot4 = np.array([base_to_vec.get(ch, [0, 0, 0, 0]) for ch in s], dtype=np.float32)
        if one_hot4.shape[0] < L:
            one_hot4 = np.vstack([one_hot4, np.zeros((L - one_hot4.shape[0], 4), dtype=np.float32)])
        else:
            one_hot4 = one_hot4[:L]
        X[i, :, :4] = one_hot4
    return X


def load_external_3pL6_a549(csv_path: Path, seq_col: str = "Nn", seq_len: int = 46) -> Tuple[np.ndarray, np.ndarray]:
    """
    CSV columns: Nn, cyt.score, nuc.score
    Build Y as [nuc, cyt] to align with model outputs [A549_nuc, A549_cyt, ...]
    """
    df = pd.read_csv(csv_path)
    need = [seq_col, "cyt.score", "nuc.score"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}. Found={df.columns.tolist()[:50]}")

    seqs = df[seq_col].astype(str).tolist()
    X = onehot6_from_seqs(seqs, L=seq_len)
    Y = df[["nuc.score", "cyt.score"]].to_numpy(np.float32)  # order: [nuc, cyt]
    return X, Y


# -------------------------
# Model (copied minimal from your code)
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
        elif model_type == 'cnn':
            self.conv1 = nn.Conv1d(in_channels=6, out_channels=256, kernel_size=8, stride=1)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(256, output_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _to_token_ids_from_onehot6(self, x6):
        bases = x6[:, :, :4]                   # (B, L, 4)
        token_ids = bases.argmax(dim=-1)       # (B, L), 0..3
        is_N = bases.sum(dim=-1) == 0          # (B, L)
        n_idx = min(self.vocab_size - 1, 4)    # reserve index for N
        token_ids = token_ids.masked_fill(is_N, n_idx)
        return token_ids.long()

    def forward(self, x):
        if self.model_type == 'lstm':
            if x.dim() == 3 and x.size(-1) == 6:
                token_ids = self._to_token_ids_from_onehot6(x)
            elif x.dim() == 2 and x.dtype in (torch.long, torch.int64):
                token_ids = x
            else:
                raise ValueError("LSTM expects (B,L,6) onehot6 or (B,L) token ids.")

            x = self.embedding(token_ids)      # (B, L, embed_dim)
            x, _ = self.lstm1(x)               # (B, L, 128)
            x, _ = self.lstm2(x)               # (B, L, 64)
            x = self.dropout1(x)
            x = self.flatten(x)                # (B, 64*L)
            x = torch.relu(self.fc1(x))
            x = self.dropout2(x)
            pred = self.fc2(x)                 # (B, output_dim)
            return pred
        else:
            x = x.permute(0, 2, 1)
            x = torch.relu(self.conv1(x))
            x = self.gap(x)
            x = self.flatten(x)
            return self.fc(x)


# -------------------------
# Metrics
# -------------------------
def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.size == 0 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


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


def eval_a549_2d(pred4: np.ndarray, true2: np.ndarray) -> Dict[str, float]:
    """
    pred4: (N,4) outputs = [A549_nuc, A549_cyt, HCT116_nuc, HCT116_cyt]
    true2: (N,2) labels  = [nuc, cyt]
    """
    pred2 = pred4[:, :2]
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
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_root", type=str, default="./models/seerr_torch_260128",
                    help="Root dir containing seed*/final_model.pth (or best_model.pth)")
    ap.add_argument("--which", type=str, default="final", choices=["final", "best"],
                    help="Evaluate final_model.pth or best_model.pth")
    ap.add_argument("--test_csv", type=str, default="./train_data_260128/3pL6-A549-T1.csv")
    ap.add_argument("--seq_col", type=str, default="Nn")
    ap.add_argument("--seq_len", type=int, default=46)

    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval_bs", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=4)

    ap.add_argument("--out_csv", type=str, default="./models/seerr_torch_260128/external_test_3pL6-A549-T1.lstm_metrics.csv")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    models_root = Path(args.models_root)
    test_csv = Path(args.test_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # load external test
    X, Y = load_external_3pL6_a549(test_csv, seq_col=args.seq_col, seq_len=args.seq_len)
    loader = DataLoader(
        SequenceDataset(X, Y),
        batch_size=args.eval_bs,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # discover seed dirs
    seed_dirs = sorted([p for p in models_root.glob("seed*") if p.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed* directories found under {models_root}")

    rows = []
    for sd in seed_dirs:
        # parse seed
        seed = None
        try:
            seed = int(sd.name.replace("seed", ""))
        except Exception:
            pass

        model_path = sd / ("final_model.pth" if args.which == "final" else "best_model.pth")
        if not model_path.exists():
            print(f"[SKIP] {sd} missing {model_path.name}")
            continue

        # build model with SAME arch as training
        model = SEERSModel(
            model_type="lstm",
            seq_length=args.seq_len,
            output_dim=4
        )
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()

        pred4, true2 = predict(model, loader, device=device)
        m = eval_a549_2d(pred4, true2)

        row = {
            "seed": seed,
            "model_path": str(model_path),
            "test_csv": str(test_csv),
            "N": int(Y.shape[0]),
            "which": args.which,
        }
        row.update(m)
        rows.append(row)

        print(f"[OK] seed={seed} | mae_mean={row['mae_mean']:.4f} r2_mean={row['r2_mean']:.4f}")

    if not rows:
        raise RuntimeError("No models evaluated. Check your models_root / file names.")

    df = pd.DataFrame(rows).sort_values(["seed"])
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
