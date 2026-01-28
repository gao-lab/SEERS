import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import time
from pathlib import Path
from scipy import stats
from collections import defaultdict
import shutil
import multiprocessing
from torch.utils.tensorboard import SummaryWriter  # TensorBoard support
import random


def set_global_seed(seed: int):
    """
    Set global random seeds for Python, NumPy, and PyTorch
    to ensure reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Custom dataset class
class SequenceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        bin_val = self.Y[idx]
        x_tensor = torch.tensor(x, dtype=torch.float32)
        return x_tensor, torch.tensor(bin_val, dtype=torch.float32)


# Model definition
class SEERSModel(nn.Module):
    def __init__(self, model_type='lstm', vocab_size=6, embed_dim=5, seq_length=150, output_dim=2):
        super(SEERSModel, self).__init__()
        self.model_type = model_type
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        if model_type == 'lstm':
            # Keras-style architecture:
            # Embedding -> LSTM(128, return_sequences)
            # -> LSTM(64, return_sequences)
            # -> Dropout -> Flatten
            # -> Dense(128) -> Dropout -> Dense(output_dim)
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
            self.lstm1 = nn.LSTM(input_size=embed_dim, hidden_size=128, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
            self.dropout1 = nn.Dropout(0.5)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * seq_length, 128)
            self.dropout2 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, output_dim)

        elif model_type == 'cnn':
            # Original CNN branch (unchanged)
            self.conv1 = nn.Conv1d(in_channels=6, out_channels=256, kernel_size=8, stride=1)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(256, output_dim)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _to_token_ids_from_onehot6(self, x6):
        """
        Convert input of shape (B, L, 6) to integer token IDs:
        - Use the first 4 dimensions as one-hot (A, C, G, T)
        - If all four are zero, treat as 'N'
        - Index for 'N' is min(vocab_size - 1, 4)
        """
        bases = x6[:, :, :4]                          # (B, L, 4)
        token_ids = bases.argmax(dim=-1)              # (B, L), values in [0, 3]
        is_N = bases.sum(dim=-1) == 0                 # All-zero rows indicate 'N'
        n_idx = min(self.vocab_size - 1, 4)           # Reserve an index for 'N'
        token_ids = token_ids.masked_fill(is_N, n_idx)
        return token_ids.long()

    def forward(self, x):
        if self.model_type == 'lstm':
            # Support two input formats:
            # - (B, L, 6): one-hot encoding (use first 4 dims)
            # - (B, L): integer token IDs
            if x.dim() == 3 and x.size(-1) == 6:
                token_ids = self._to_token_ids_from_onehot6(x)
            elif x.dim() == 2 and x.dtype in (torch.long, torch.int64):
                token_ids = x
            else:
                raise ValueError(
                    "LSTM+Embedding expects input of shape (B, L, 6) "
                    "with one-hot encoding or (B, L) integer token IDs."
                )

            x = self.embedding(token_ids)             # (B, L, embed_dim)
            x, _ = self.lstm1(x)                      # (B, L, hidden)
            x, _ = self.lstm2(x)                      # (B, L, hidden)
            x = self.dropout1(x)
            x = self.flatten(x)                       # (B, L*hidden)
            x = torch.relu(self.fc1(x))
            x = self.dropout2(x)
            pred_score = self.fc2(x)

        else:  # CNN branch
            x = x.permute(0, 2, 1)                    # (B, C=6, L)
            x = torch.relu(self.conv1(x))
            x = self.gap(x)
            x = self.flatten(x)
            pred_score = self.fc(x)

        return pred_score


def train_model(model, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, save_path, config):
    """
    Train the model and save checkpoints.

    Args:
        model: initialized model instance
        X_train, Y_train: training data (sequences and labels), shape (N, seq_len, channels)
        X_valid, Y_valid: validation data
        X_test, Y_test: test data
        save_path: directory for saving models
        config: dict of training configurations
    """
    # Use seed-specific run subdirectory
    run_seed = int(config.get('seed', 42))
    save_path = Path(save_path) / f"seed{run_seed}"
    save_path.mkdir(exist_ok=True, parents=True)

    final_model_path = save_path / "final_model.pth"
    best_model_path = save_path / "best_model.pth"

    # Set global seed
    set_global_seed(run_seed)

    # Device
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # If final model exists, load and skip training, then return test predictions
    if final_model_path.exists():
        print(f"Found pretrained model {final_model_path}, skipping training")
        model.load_state_dict(torch.load(final_model_path))

        # Prepare test dataset/loader
        class TestDataset(Dataset):
            def __init__(self, X, Y):
                self.X = X
                self.Y = Y

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                x = self.X[idx]
                bin_val = self.Y[idx]
                x_tensor = torch.tensor(x, dtype=torch.float32)
                return x_tensor, torch.tensor(bin_val, dtype=torch.float32)

        test_dataset = TestDataset(X_test, Y_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.get('valid_batch_size', 4096),
            shuffle=False,
            num_workers=config.get('valid_workers', 4),
            pin_memory=True
        )

        # Predict on test set
        model.eval()
        test_preds = []
        test_labels = []

        device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        with torch.no_grad():
            for X, bin_val in test_loader:
                X = X.to(device)
                pred_score = model(X)
                test_preds.append(pred_score.cpu().numpy())
                test_labels.append(bin_val.cpu().numpy())

        test_preds = np.concatenate(test_preds)
        test_labels = np.concatenate(test_labels)

        return test_preds, test_labels

    # Otherwise, start training
    print("No pretrained model found, starting training")

    # Save config
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Set seeds again (redundant but kept as-is)
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare datasets
    train_dataset = SequenceDataset(X_train, Y_train)
    valid_dataset = SequenceDataset(X_valid, Y_valid)
    test_dataset = SequenceDataset(X_test, Y_test)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        num_workers=config['train_workers'],
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['valid_batch_size'],
        shuffle=False,
        num_workers=config['valid_workers'],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['valid_batch_size'],
        shuffle=False,
        num_workers=config['valid_workers'],
        pin_memory=True
    )

    # Initialize optimizer
    min_lr = config['max_lr'] / config['div_factor']
    if config['optimizer'] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=min_lr,
            betas=(0.9, 0.999),
            eps=1e-07,
            amsgrad=False,
            weight_decay=0.0
        )
    elif config['optimizer'] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=min_lr, weight_decay=config['weight_decay'])
    elif config['optimizer'] == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=min_lr, weight_decay=config['weight_decay'])

    # Loss function: MAE
    criterion = nn.L1Loss()

    # Learning rate scheduler
    if config['scheduler'] == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['max_lr'],
            steps_per_epoch=len(train_loader),
            epochs=config['epoch_num'],
            pct_start=config['pct_start'],
            div_factor=config['div_factor']
        )
    else:
        scheduler = None

    # TensorBoard writer
    writer = SummaryWriter(save_path / "logs")

    # Early stopping config
    patience = config.get('patience', 10)
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    early_stop = False

    # Training loop
    train_history = defaultdict(list)
    for epoch in range(config['epoch_num']):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        # Training phase
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for batch_idx, (X, bin_val) in enumerate(train_loader):
            X = X.to(device)
            bin_val = bin_val.to(device)
            optimizer.zero_grad()

            # Forward pass
            pred_score = model(X)

            # Compute loss
            loss = criterion(pred_score, bin_val)

            # Backpropagation
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            epoch_loss += loss.item()

        # Average training loss
        avg_train_loss = epoch_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X, bin_val in valid_loader:
                X = X.to(device)
                bin_val = bin_val.to(device)

                pred_score = model(X)
                loss = criterion(pred_score, bin_val)

                val_loss += loss.item()
                all_preds.append(pred_score.cpu().numpy())
                all_labels.append(bin_val.cpu().numpy())

        # Average validation loss
        avg_val_loss = val_loss / len(valid_loader)

        # Validation metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Pearson and Spearman correlations for each output
        pearson_nuc = np.corrcoef(all_preds[:, 0], all_labels[:, 0])[0, 1]
        spearman_nuc = stats.spearmanr(all_preds[:, 0], all_labels[:, 0]).correlation
        pearson_cyt = np.corrcoef(all_preds[:, 1], all_labels[:, 1])[0, 1]
        spearman_cyt = stats.spearmanr(all_preds[:, 1], all_labels[:, 1]).correlation

        # Record history
        train_history['epoch'].append(epoch)
        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_pearson_nuc'].append(pearson_nuc)
        train_history['val_spearman_nuc'].append(spearman_nuc)
        train_history['val_pearson_cyt'].append(pearson_cyt)
        train_history['val_spearman_cyt'].append(spearman_cyt)

        # TensorBoard logging
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Metrics/pearson_nuc', pearson_nuc, epoch)
        writer.add_scalar('Metrics/spearman_nuc', spearman_nuc, epoch)
        writer.add_scalar('Metrics/pearson_cyt', pearson_cyt, epoch)
        writer.add_scalar('Metrics/spearman_cyt', spearman_cyt, epoch)

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model found, val_loss={best_val_loss:.4f}, saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve, patience: {patience_counter}/{patience}")

            # Check early stopping condition
            if patience_counter >= patience:
                early_stop = True
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")

        # Save a checkpoint every 10 epochs
        if epoch % 10 == 0:
            epoch_model_path = save_path / f"model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), epoch_model_path)

        # Print progress
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{config['epoch_num']} | "
            f"Time: {epoch_time:.1f}s | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Pearson (nuc): {pearson_nuc:.4f} | "
            f"Pearson (cyt): {pearson_cyt:.4f}"
        )

    # After training
    print(f"Training finished. Best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f}")

    # Load best model
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model: {best_model_path}")
    else:
        print("Warning: best model not found; using final model weights")

    # Save final model
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

    # Save training history
    with open(save_path / "train_history.json", "w") as f:
        json.dump(train_history, f, indent=4)

    # Testing phase
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for X, bin_val in test_loader:
            X = X.to(device)
            pred_score = model(X)
            test_preds.append(pred_score.cpu().numpy())
            test_labels.append(bin_val.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)

    # Test metrics
    test_mae_nuc = np.mean(np.abs(test_preds[:, 0] - test_labels[:, 0]))
    test_mae_cyt = np.mean(np.abs(test_preds[:, 1] - test_labels[:, 1]))
    test_pearson_nuc = np.corrcoef(test_preds[:, 0], test_labels[:, 0])[0, 1]
    test_pearson_cyt = np.corrcoef(test_preds[:, 1], test_labels[:, 1])[0, 1]

    # Save test results
    test_results = {
        'mae_nuc': test_mae_nuc,
        'mae_cyt': test_mae_cyt,
        'pearson_nuc': test_pearson_nuc,
        'pearson_cyt': test_pearson_cyt,
        'predictions': test_preds.tolist(),
        'labels': test_labels.tolist()
    }

    print(f"\nTest results - Nuc: MAE={test_mae_nuc:.4f}, Pearson={test_pearson_nuc:.4f}")
    print(f"Test results - Cyt: MAE={test_mae_cyt:.4f}, Pearson={test_pearson_cyt:.4f}")

    return test_preds, test_labels


def qqplot_r2(y_true, y_pred, xlabel='Observed', ylabel='Predicted', outfile='qqplot.png', figsize=(12, 12)):
    """
    Plot a QQ-like scatter (predicted vs observed) and annotate Pearson R^2.
    y_true, y_pred: 1D numpy arrays
    """
    # Compute R^2
    slope, intercept, r_value, p_value, stderr = linregress(y_true, y_pred)
    r2 = r_value ** 2

    # Plot
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.scatter(y_true, y_pred, alpha=0.6, s=2)

    # 45-degree reference line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'QQ-plot (Pearson R² = {r2:.3f})')

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def prepare_legnet_input_multicell(csv_path, seq_col="Nn", seqsize=46):
    """
    Required columns:
      Nn, cyt.score.A549, nuc.score.A549, cyt.score.HCT116, nuc.score.HCT116

    Returns:
      X: (N, seqsize, 6)
      Y: (N, 4) in the order:
         [nuc.A549, cyt.A549, nuc.HCT116, cyt.HCT116]
    """
    df = pd.read_csv(csv_path)

    need_cols = [
        seq_col,
        "nuc.score.A549", "cyt.score.A549",
        "nuc.score.HCT116", "cyt.score.HCT116"
    ]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}. Available columns: {list(df.columns)}")

    seqs = df[seq_col].astype(str).tolist()

    base_to_vec = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }

    X_final = np.zeros((len(seqs), seqsize, 6), dtype=np.float32)

    for i, seq in enumerate(seqs):
        seq = seq.upper()
        one_hot4 = np.array([base_to_vec.get(b, [0, 0, 0, 0]) for b in seq], dtype=np.float32)

        if one_hot4.shape[0] < seqsize:
            pad = np.zeros((seqsize - one_hot4.shape[0], 4), dtype=np.float32)
            one_hot4 = np.vstack([one_hot4, pad])
        else:
            one_hot4 = one_hot4[:seqsize]

        full6 = np.zeros((seqsize, 6), dtype=np.float32)
        full6[:, :4] = one_hot4
        X_final[i] = full6

    # Fixed label order (4 dimensions)
    Y = df[["nuc.score.A549", "cyt.score.A549", "nuc.score.HCT116", "cyt.score.HCT116"]].to_numpy(dtype=np.float32)
    return X_final, Y


def evaluate_multicell(model, X, Y, config, out_dir: Path, tag: str):
    """
    Y: (N, 4) = [nuc.A549, cyt.A549, nuc.HCT116, cyt.HCT116]
    """
    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    ds = SequenceDataset(X, Y)
    loader = DataLoader(
        ds,
        batch_size=config.get('valid_batch_size', 4096),
        shuffle=False,
        num_workers=config.get('valid_workers', 4),
        pin_memory=True
    )

    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yhat = model(xb)
            preds.append(yhat.cpu().numpy())
            labels.append(yb.numpy())

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    metrics = {
        "n": int(labels.shape[0]),
        "A549_pearson_nuc": corr(preds[:, 0], labels[:, 0]),
        "A549_pearson_cyt": corr(preds[:, 1], labels[:, 1]),
        "HCT116_pearson_nuc": corr(preds[:, 2], labels[:, 2]),
        "HCT116_pearson_cyt": corr(preds[:, 3], labels[:, 3]),
        "A549_spearman_nuc": float(stats.spearmanr(preds[:, 0], labels[:, 0]).correlation),
        "A549_spearman_cyt": float(stats.spearmanr(preds[:, 1], labels[:, 1]).correlation),
        "HCT116_spearman_nuc": float(stats.spearmanr(preds[:, 2], labels[:, 2]).correlation),
        "HCT116_spearman_cyt": float(stats.spearmanr(preds[:, 3], labels[:, 3]).correlation),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{tag}_pred.npy", preds)
    np.save(out_dir / f"{tag}_true.npy", labels)
    with open(out_dir / f"{tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"[{tag}] n={metrics['n']} | "
        f"A549 pearson nuc={metrics['A549_pearson_nuc']:.4f} cyt={metrics['A549_pearson_cyt']:.4f} | "
        f"HCT116 pearson nuc={metrics['HCT116_pearson_nuc']:.4f} cyt={metrics['HCT116_pearson_cyt']:.4f}"
    )
    return metrics


def maintrain(seed):
    # ========== Basic configuration ==========
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_type = 'lstm'
    seq_size = 46

    model = SEERSModel(
        model_type=model_type,
        seq_length=seq_size,
        output_dim=4
    )

    # Your model save directory (seed-specific subfolders)
    save_path = './models/seerr_torch_260106/'

    config = {
        'gpu': 0,
        'seed': seed,
        'train_batch_size': 16348,
        'valid_batch_size': 4096,
        'train_workers': multiprocessing.cpu_count() // 2,
        'valid_workers': multiprocessing.cpu_count() // 2,
        'max_lr': 0.5**9,
        'div_factor': 25,
        'pct_start': 0.3,
        'epoch_num': 1000,
        'optimizer': "adam",
        'scheduler': "onecycle",
        'weight_decay': 0.0001,
        'patience': 100
    }
    set_global_seed(config['seed'])

    # ========== Load current train/val/test splits ==========
    # Current directory structure:
    # ./251230_train_set/train_set.csv
    # ./251230_train_set/val_set.csv
    # ./251230_train_set/test_set.csv
    base_dir = Path("./train_data_260106")
    train_csv = base_dir / "train_set.csv"
    val_csv = base_dir / "val_set.csv"
    test_csv = base_dir / "test_set.csv"

    X_train, Y_train = prepare_legnet_input_multicell(train_csv, seq_col="Nn", seqsize=seq_size)
    X_valid, Y_valid = prepare_legnet_input_multicell(val_csv, seq_col="Nn", seqsize=seq_size)
    X_test, Y_test = prepare_legnet_input_multicell(test_csv, seq_col="Nn", seqsize=seq_size)

    print("Train:", X_train.shape, Y_train.shape)  # Y should be (N, 4)
    print("Val  :", X_valid.shape, Y_valid.shape)
    print("Test :", X_test.shape, Y_test.shape)

    # ========== Train ==========
    # Note: Passing test data here is only to match the original train_model signature.
    # The per-cell evaluation is performed separately after training.
    Y_test_pred, Y_test_true = train_model(
        model,
        X_train, Y_train,
        X_valid, Y_valid,
        X_test, Y_test,
        save_path, config
    )

    # ========== Load final model and evaluate separately for A549/HCT116 ==========
    run_seed = int(config.get('seed', 42))
    run_dir = Path(save_path) / f"seed{run_seed}"
    final_model_path = run_dir / "final_model.pth"
    if final_model_path.exists():
        model.load_state_dict(torch.load(final_model_path, map_location="cpu"))

    run_dir = Path(save_path) / f"seed{int(config.get('seed', 42))}"
    out_eval_dir = run_dir / "eval_by_cell"
    evaluate_multicell(model, X_test, Y_test, config, out_eval_dir, tag="TEST_MULTICELL")


if __name__ == "__main__":
    randomseedslist = [42]
    for seed in randomseedslist:
        maintrain(seed)
