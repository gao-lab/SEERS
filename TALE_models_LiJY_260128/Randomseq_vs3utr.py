# -*- coding: utf-8 -*-
import os
import itertools
import pdb

import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde
from scipy.spatial.distance import pdist, squareform

# ---------------------------
# k-mer utilities
# ---------------------------

def _kmer_vocab(k: int) -> List[str]:
    alphabet = ['A', 'C', 'G', 'T']
    return [''.join(p) for p in itertools.product(alphabet, repeat=k)]

def _seq_to_kmer_counts(seq: str, k: int) -> Counter:
    s = seq.upper()
    kmers = [s[i:i+k] for i in range(len(s)-k+1)]
    kmers = [k for k in kmers if set(k) <= set("ACGT")]
    return Counter(kmers)

def sequences_to_kmer_matrix(seqs: List[str], k: int, norm: str = "l1") -> Tuple[np.ndarray, List[str]]:
    """
    Return matrix shape (n_seq, 4**k) of k-mer frequencies.
    norm: "l1" for frequency (sum=1), None for raw counts.
    """
    vocab = _kmer_vocab(k)
    idx = {w:i for i,w in enumerate(vocab)}
    X = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for r, seq in enumerate(seqs):
        cnt = _seq_to_kmer_counts(seq, k)
        for w, c in cnt.items():
            if w in idx: X[r, idx[w]] = c
    if norm == "l1":
        X = normalize(X, norm="l1", copy=False)
    return X, vocab

# ---------------------------
# Metrics
# ---------------------------

def js_divergence_hist(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Jensen–Shannon divergence between pooled k-mer distributions.
    """
    p = X.sum(axis=0); q = Y.sum(axis=0)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)

def _pairwise_rbf_kernel(D2: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * D2)

def mmd_rbf(X: np.ndarray, Y: np.ndarray, bandwidth: str = "median") -> float:
    """
    Unbiased MMD with RBF kernel. Smaller = closer distributions.
    bandwidth: "median" (median heuristic) or float value of gamma.
    """
    Z = np.vstack([X, Y])
    D2 = pairwise_distances(Z, metric="sqeuclidean")
    if bandwidth == "median":
        # median of non-zero distances
        tri = D2[np.triu_indices_from(D2, k=1)]
        med = np.median(tri[tri > 0])
        gamma = 1.0 / (2.0 * med + 1e-12)
    else:
        gamma = float(bandwidth)

    n, m = len(X), len(Y)
    K = _pairwise_rbf_kernel(D2, gamma)

    Kxx = K[:n, :n]
    Kyy = K[n:, n:]
    Kxy = K[:n, n:]

    # Unbiased estimator
    mmd = (Kxx.sum() - np.trace(Kxx)) / (n*(n-1)+1e-12) \
        + (Kyy.sum() - np.trace(Kyy)) / (m*(m-1)+1e-12) \
        - 2.0 * Kxy.mean()
    return float(mmd)

def nn_coverage(X_ref: np.ndarray, X_cand: np.ndarray, quantiles=(0.25, 0.50, 0.75)) -> Dict[float, float]:
    """
    For each point in X_ref, compute its NN distance to other X_ref points.
    Use the chosen quantiles of that scale as radius r, and report
    fraction of X_ref whose nearest neighbor in X_cand is within r.
    Returns dict {q: coverage}.
    """
    # Reference internal scale
    nbr_ref = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X_ref)
    d_ref, _ = nbr_ref.kneighbors(X_ref)  # distances to self + nearest other
    ref_scale = d_ref[:, 1]  # skip self (0)

    # Candidate distances
    nbr_cand = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(X_cand)
    d_cand, _ = nbr_cand.kneighbors(X_ref)
    d_cand = d_cand.ravel()

    out = {}
    for q in quantiles:
        r = np.quantile(ref_scale, q)
        out[q] = float(np.mean(d_cand <= r))
    return out

# ---------------------------
# Embedding & plotting
# ---------------------------

def coembed_2d(X: np.ndarray, y: np.ndarray, method: str = "umap", random_state: int = 42) -> np.ndarray:
    """
    X: (n_samples, n_features); y: labels (unused except for potential stratified diagnostics)
    method: "umap" or "pca"
    """
    # Pre-PCA to 50 dims for stability/speed
    d = min(50, X.shape[1], X.shape[0]-1) if X.shape[0] > 2 else min(2, X.shape[1])
    Xp = PCA(n_components=d, random_state=random_state).fit_transform(X)

    if method == "umap" and HAS_UMAP and Xp.shape[0] >= 10:
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="euclidean",
                            random_state=random_state)
        Z = reducer.fit_transform(Xp)
    else:
        # fall back to PCA-2D
        Z = PCA(n_components=2, random_state=random_state).fit_transform(Xp)
    return Z

def _kde_contour(ax, pts: np.ndarray, levels=(0.5, 0.8, 0.95), label=None, linewidths=1.5):
    if len(pts) < 100:
        ax.scatter(pts[:,0], pts[:,1], s=4, alpha=0.4, label=label)
        return
    kde = gaussian_kde(pts.T)
    xmin, ymin = pts.min(axis=0) - 0.5
    xmax, ymax = pts.max(axis=0) + 0.5
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    # normalize to cumulative levels
    zsort = np.sort(zz.ravel())[::-1]
    cumsum = np.cumsum(zsort)
    cumsum /= cumsum[-1]
    thr = {lev: zsort[np.searchsorted(cumsum, lev)] for lev in levels}
    cs = ax.contour(xx, yy, zz, levels=[thr[l] for l in levels],
                    linewidths=linewidths)
    if label: cs.collections[0].set_label(label)

def plot_coembedding(Z: np.ndarray, labels: np.ndarray, outpath: str,
                     title: str = "", show_points=False):
    """
    Draw separate density contours for two classes to avoid overplotting.
    """
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    grp0 = Z[labels==0]; grp1 = Z[labels==1]

    # density contours
    _kde_contour(ax, grp0, label="N45 random", linewidths=2.0)
    _kde_contour(ax, grp1, label="Human 3'UTR (45nt)", linewidths=2.0)

    if show_points:
        ax.scatter(grp0[:,0], grp0[:,1], s=2, alpha=0.15)
        ax.scatter(grp1[:,0], grp1[:,1], s=4, alpha=0.35)

    ax.set_xlabel("Embed-1")
    ax.set_ylabel("Embed-2")
    ax.set_title(title or "Co-embedding of k-mer features")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

# ---------------------------
# Main entry
# ---------------------------

def analyze_kmer_overlap(utrseqlist: List[str],
                         randomseqlist: List[str],
                         k_list=(3,4,5),
                         embed_method="umap",
                         outdir="./kmer_overlap",
                         random_state=42) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)
    results = []

    for k in k_list:
        X_utr, _ = sequences_to_kmer_matrix(utrseqlist, k=k, norm="l1")
        X_rand, _ = sequences_to_kmer_matrix(randomseqlist, k=k, norm="l1")

        # Co-embedding (shared space)
        X = np.vstack([X_rand, X_utr])
        y = np.array([0]*len(X_rand) + [1]*len(X_utr), dtype=np.int8)

        Z = coembed_2d(X, y, method=embed_method, random_state=random_state)

        # ---- save coords + labels ----
        out_csv = os.path.join(outdir, f"umap_coords_k{k}_{embed_method}.csv")
        df_coords = pd.DataFrame({
            "embed1": Z[:, 0],
            "embed2": Z[:, 1],
            "label": y,
            "source": np.where(y == 0, "random", "utr"),
        })
        df_coords.to_csv(out_csv, index=False)

        # (optional) also save as npz for exact float preservation
        out_npz = os.path.join(outdir, f"umap_coords_k{k}_{embed_method}.npz")
        np.savez_compressed(out_npz, Z=Z.astype(np.float32), y=y)

        # plot
        # fig_path = os.path.join(outdir, f"coembed_k{k}_{embed_method}.png")
        # plot_coembedding(Z, y, fig_path, title=f"Co-embedding (k={k}, method={embed_method})")





def load3utr():
    '''

    :return:
    '''
    fasta_path = '../seerr_data/hg38_3utr.fa'
    seq_length = 45
    utr_list = []
    with open(fasta_path, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):  # 新的序列头
                if seq:
                    utr_list.append(seq[:seq_length].upper())
                seq = ''
            else:
                seq += line
        # 最后一条序列
        if seq:
            utr_list.append(seq[:seq_length].upper())
    return utr_list

def main():
    '''

    :return:
    '''


    utrseqlist =  load3utr()  # placeholder
    # seerrlist = pd.read_csv('../seerr_data/L5_log2expression_train_220528.tsv',sep='\t')['Nn'].tolist()+pd.read_csv('../seerr_data/L5_log2expression_val_220528.tsv',sep='\t')['Nn'].tolist()

    # new version
    seerrlist = pd.read_csv('train_data_260106/train_set.csv',sep=',')['Nn'].tolist()+pd.read_csv('train_data_260106/val_set.csv',sep=',')['Nn'].tolist()

    # Remove the above lines and pass your real data to analyze_kmer_overlap
    df = analyze_kmer_overlap(
        utrseqlist=utrseqlist,
        randomseqlist=seerrlist,
        k_list=(4,6,8),
        embed_method="umap",
        outdir="./kmer_overlap"
    )


if __name__ == "__main__":
    main()


