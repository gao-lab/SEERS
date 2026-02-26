## SEERS: Selective Enrichment of Episomes with Random Sequences

**A systematic delineation of 3′ UTR regulatory elements and their contextual associations.**

---

### 📖 Introduction

SEERS (Selective Enrichment of Episomes with Random Sequences) is an **improved MPRA (Massively Parallel Reporter Assay) workflow** for large-scale functional interrogation of DNA sequences while **minimizing transfection-associated perturbation and toxicity**.

The key design of SEERS is the use of an **EBNA1/OriP-based episomal vector**, which enables **delayed cell collection** after transfection. This reduces acute cellular stress and nonspecific transcriptional effects caused by transfection, improving robustness and reproducibility of sequence–function measurements.

Although SEERS is applied here to **3′ UTR** regulation, the workflow is **generalizable in principle to other DNA sequence types**, including promoters, enhancers, UTRs, or other cis-regulatory elements.

This GitHub repository supports our manuscript in which we assayed ~**2 million** **45-nt random 3′ UTR** sequences using the SEERS workflow to quantify their regulatory effects on:
1) **gene expression output**, and  
2) **nuclear–cytoplasmic partitioning**.

These measurements were further used to train a deep learning model named **TALE** for predicting regulatory activity, and to systematically identify **short 3′ UTR regulatory elements (2–8-mers)** and characterize their functional associations.


---

### 🛠️ Key Features

* **Improved MPRA workflow:** Uses an **EBNA1/OriP episomal vector** to allow **delayed sampling**, reducing transfection-induced perturbation/toxicity.
* **Massively parallel functional assay:** Quantifies the effects of **millions of sequences** in a single framework.
* **Multi-compartment readout:** Captures regulation of **expression** and **nuclear–cytoplasmic localization** (DNA, Cytoplasm, Nucleus as supported by this repo’s analysis scripts).
* **Motif discovery:** Built-in k-mer profiling to identify and quantify **2–8-mer** regulatory elements associated with measured activity.
* **Deep learning ready:** Includes the **TALE** model code/weights and utilities for training and evaluation, plus notebooks for SNP effect prediction.


---

### 📂 Repository Structure

#### 1. Data Processing & k-mer Analyses

`SEERS_data_process_LiangN/`

* `Nn_pp.R`: Extract and count N45 sequences from merged FASTQ files.
* `Nn_pp_pool.R`: Aggregate counts across multiple biological and technical replicates.
* `combine_dna_cyt_nuc.R`: Compute enrichment scores (SEERS data) by normalizing Cyt/Nuc counts against DNA input.
* `kmer_profiling.R`: Statistical testing of k-mer correlations with regulatory activity.

#### 2. Deep Learning Models

`TALE_models_LiJY_260128/`

* Contains the architecture and weights for the **TALE models**.
* Also includes scripts and utilities for model training, evaluation/testing, and other related works.

---

### 🚀 Getting Started

#### Prerequisites

* **Bioinformatics Tools:** [NGmerge](https://github.com/jsh58/NGmerge)
* **Language Environment:** R (>= 4.0), Python (>= 3.9)

#### Step 1: Pre-processing

Merge the paired-end sequencing data:

```bash
./NGmerge -d -1 read1.fq.gz -2 read2.fq.gz -o merged.fq.gz

```

#### Step 2: Training Data

Download the full training dataset from Zenodo:

🔗 [https://doi.org/10.5281/zenodo.18737939](https://doi.org/10.5281/zenodo.18737939)

---

### 📝 Change Log

| Date | Version/Update | Description |
| --- | --- | --- |
| **2026-01-28** | v2.0 | Updated scripts for revised manuscript and new datasets. |
| **2025-04-21** | v1.2 | Added `kmer_motif.ipynb` and `N45_dissect.ipynb`. |
| **2024-12-08** | v1.1 | Added `TALE_SNP_effect.ipynb`. |
| **2024-08-14** | v1.0 | TF 2.16 compatibility fix & Refactored Notebooks. |

---

