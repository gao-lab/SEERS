#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Data
# -------------------------
class SeqDS(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype(np.float32)  # (N,L,6)
        self.Y = Y.astype(np.float32)  # (N,4)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


def load_multicell(csv_path, seq_col="Nn", L=46):
    df = pd.read_csv(csv_path)
    need = [seq_col, "nuc.score.A549", "cyt.score.A549", "nuc.score.HCT116", "cyt.score.HCT116"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing cols: {miss}")

    seqs = df[seq_col].astype(str).tolist()
    base = {'A': [1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'T':[0,0,0,1], 'N':[0,0,0,0]}
    X = np.zeros((len(seqs), L, 6), dtype=np.float32)

    for i, s in enumerate(seqs):
        s = s.upper()
        oh4 = np.array([base.get(ch, [0,0,0,0]) for ch in s], dtype=np.float32)
        if oh4.shape[0] < L:
            oh4 = np.vstack([oh4, np.zeros((L-oh4.shape[0], 4), np.float32)])
        else:
            oh4 = oh4[:L]
        X[i, :, :4] = oh4

    Y = df[["nuc.score.A549", "cyt.score.A549", "nuc.score.HCT116", "cyt.score.HCT116"]].to_numpy(np.float32)
    return X, Y


# -------------------------
# Model: Conv1d + GlobalMaxPool + 1-layer MLP
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
        # x: (B,L,6) -> (B,6,L)
        x = x.permute(0, 2, 1)
        x = self.act(self.conv(x))
        x = self.gmp(x).squeeze(-1)   # (B, C)
        x = self.drop(x)
        x = self.act(self.mlp(x))
        return self.head(x)


# -------------------------
# Train / Eval
# -------------------------
def pearson_r(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, T = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yhat = model(xb).detach().cpu().numpy()
        P.append(yhat)
        T.append(yb.numpy())
    return np.concatenate(P, 0), np.concatenate(T, 0)


def train_one_kernel(
    kernel_size,
    Xtr, Ytr, Xva, Yva, Xte, Yte,
    device,
    seed=42,
    cnn_out=256,
    mlp_hidden=128,
    dropout=0.0,
    epochs=300,
    patience=40,
    lr=1e-3,
    weight_decay=1e-4,
    train_bs=16384,
    eval_bs=4096,
    workers=4,
    save_path: str = None,   # <-- 新增
):
    set_seed(seed)
    model = CNN1(kernel_size, cnn_out=cnn_out, mlp_hidden=mlp_hidden, dropout=dropout).to(device)

    tr_loader = DataLoader(SeqDS(Xtr, Ytr), batch_size=train_bs, shuffle=True,
                           num_workers=workers, pin_memory=True)
    va_loader = DataLoader(SeqDS(Xva, Yva), batch_size=eval_bs, shuffle=False,
                           num_workers=workers, pin_memory=True)
    te_loader = DataLoader(SeqDS(Xte, Yte), batch_size=eval_bs, shuffle=False,
                           num_workers=workers, pin_memory=True)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    best_val = np.inf
    best_state = None
    bad = 0

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        vp, vt = predict(model, va_loader, device)
        val_mae = float(np.mean(np.abs(vp - vt)))

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    tp, tt = predict(model, te_loader, device)
    r2 = []
    for j in range(4):
        r = pearson_r(tp[:, j], tt[:, j])
        r2.append(r*r if np.isfinite(r) else np.nan)

    # ---- 保存 checkpoint（best_state + meta）----
    if save_path is not None:
        ckpt = {
            "model_class": "CNN1",
            "kernel_size": int(kernel_size),
            "seed": int(seed),
            "cnn_out": int(cnn_out),
            "mlp_hidden": int(mlp_hidden),
            "dropout": float(dropout),
            "seq_len": int(Xtr.shape[1]),
            "in_ch": int(Xtr.shape[2]),
            "best_val_MAE": float(best_val),
            "test_R2": [float(x) if np.isfinite(x) else np.nan for x in r2],
            "state_dict": best_state if best_state is not None else model.state_dict(),
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(ckpt, save_path)

    return best_val, r2


# -------------------------
# Plot (error bars)
# -------------------------
def plot_r2_errorbar(df_agg, out_png):
    ks = df_agg["kernel_size"].to_numpy()

    keys = ["A549_nuc_R2", "A549_cyt_R2", "HCT116_nuc_R2", "HCT116_cyt_R2"]
    labels = ["A549 nuc", "A549 cyt", "HCT116 nuc", "HCT116 cyt"]

    plt.figure(figsize=(7.6, 4.8))
    for k, lab in zip(keys, labels):
        mu = df_agg[f"{k}_mean"].to_numpy()
        sd = df_agg[f"{k}_std"].to_numpy()
        plt.errorbar(ks, mu, yerr=sd, marker="o", capsize=3, label=lab)

    plt.xlabel("Kernel size")
    plt.ylabel("Predictive performance (R²)")
    plt.ylim(0, 1.0)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_loss_errorbar(df_agg, out_png):
    ks = df_agg["kernel_size"].to_numpy()
    mu = df_agg["best_val_MAE_mean"].to_numpy()
    sd = df_agg["best_val_MAE_std"].to_numpy()

    plt.figure(figsize=(7.0, 4.6))
    plt.errorbar(ks, mu, yerr=sd, marker="o", capsize=3)
    plt.xlabel("Kernel size")
    plt.ylabel("Best val MAE (loss)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def aggregate_over_seeds(df_all: pd.DataFrame) -> pd.DataFrame:
    metrics = ["best_val_MAE", "A549_nuc_R2", "A549_cyt_R2", "HCT116_nuc_R2", "HCT116_cyt_R2"]

    out = (
        df_all
        .groupby("kernel_size")[metrics]
        .agg(["mean", "std"])
    )

    # flatten columns: (metric, stat) -> metric_stat
    out.columns = [f"{m}_{stat}" for (m, stat) in out.columns.to_list()]

    out = out.reset_index()  # kernel_size becomes a normal column
    return out.sort_values("kernel_size").reset_index(drop=True)


# -------------------------
# Main
# -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./train_data_260106")
    ap.add_argument("--out_dir", type=str, default="./cnn1_kernel_sweep_out_seeds")
    ap.add_argument("--seq_col", type=str, default="Nn")
    ap.add_argument("--seq_len", type=int, default=46)

    ap.add_argument("--kernel_min", type=int, default=2)
    ap.add_argument("--kernel_max", type=int, default=11)

    # 3 seeds by default
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 5143])

    ap.add_argument("--cnn_out", type=int, default=256)
    ap.add_argument("--mlp_hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--train_bs", type=int, default=16384)
    ap.add_argument("--eval_bs", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, Ytr = load_multicell(data_dir / "train_set.csv", seq_col=args.seq_col, L=args.seq_len)
    Xva, Yva = load_multicell(data_dir / "val_set.csv",   seq_col=args.seq_col, L=args.seq_len)
    Xte, Yte = load_multicell(data_dir / "test_set.csv",  seq_col=args.seq_col, L=args.seq_len)

    all_rows = []
    models_dir = out_dir / "models"
    for seed in args.seeds:
        print(f"\n==== Seed {seed} ====")
        seed_dir = models_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for ks in range(args.kernel_min, args.kernel_max + 1):
            save_path = str(seed_dir / f"ks{ks}.pt")
            best_val, r2 = train_one_kernel(
                ks, Xtr, Ytr, Xva, Yva, Xte, Yte,
                device=device,
                seed=seed,
                cnn_out=args.cnn_out,
                mlp_hidden=args.mlp_hidden,
                dropout=args.dropout,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                weight_decay=args.weight_decay,
                train_bs=args.train_bs,
                eval_bs=args.eval_bs,
                workers=args.workers,
                save_path=save_path,  # <-- 新增
            )
            row = {
                "kernel_size": ks,
                "seed": seed,
                "best_val_MAE": best_val,
                "A549_nuc_R2": r2[0],
                "A549_cyt_R2": r2[1],
                "HCT116_nuc_R2": r2[2],
                "HCT116_cyt_R2": r2[3],
            }
            rows.append(row)
            print(f"[seed={seed} ks={ks}] val_MAE={best_val:.4f} | "
                  f"R2: {r2[0]:.4f},{r2[1]:.4f},{r2[2]:.4f},{r2[3]:.4f}")

        df_seed = pd.DataFrame(rows).sort_values("kernel_size")
        df_seed.to_csv(out_dir / f"metrics_kernel_sweep.seed{seed}.csv", index=False)
        all_rows.extend(rows)

    df_all = pd.DataFrame(all_rows).sort_values(["kernel_size", "seed"])
    df_all.to_csv(out_dir / "metrics_kernel_sweep.all_seeds.csv", index=False)

    df_agg = aggregate_over_seeds(df_all)
    df_agg.to_csv(out_dir / "metrics_kernel_sweep.agg_mean_std.csv", index=False)

    plot_r2_errorbar(df_agg, out_dir / "kernel_sweep_R2_4outputs.errorbar.png")
    plot_loss_errorbar(df_agg, out_dir / "kernel_sweep_valMAE.errorbar.png")

    print("\nSaved:")
    print(" ", out_dir / "metrics_kernel_sweep.all_seeds.csv")
    for seed in args.seeds:
        print(" ", out_dir / f"metrics_kernel_sweep.seed{seed}.csv")
    print(" ", out_dir / "metrics_kernel_sweep.agg_mean_std.csv")
    print(" ", out_dir / "kernel_sweep_R2_4outputs.errorbar.png")
    print(" ", out_dir / "kernel_sweep_valMAE.errorbar.png")


if __name__ == "__main__":
    main()
