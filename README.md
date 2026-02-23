## SEERS: Selective Enrichment of Episomes with Random Sequences

**A systematic delineation of 3′ UTR regulatory elements and their contextual associations.**

---

### 📖 Introduction

SEERS (Selective Enrichment of Episomes with Random Sequences) is a high-throughput framework designed to dissect the regulatory landscape of **3′ UTRs**. By leveraging random sequence libraries and episomal enrichment, this pipeline allows for the systematic identification of regulatory motifs and the quantification of their functional impact on gene expression.

### 🛠️ Key Features

* **Massively Parallel Analysis:** Quantifies millions of random or specific sequences simultaneously.
* **Multi-Compartment Profiling:** Supports data integration from DNA, Cytoplasm, and Nucleus.
* **Motif Discovery:** Built-in tools for k-mer profiling and SNP effect prediction.
* **Deep Learning Ready:** Includes pre-trained TALE models for regulatory activity prediction.

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
* **Language Environment:** R (>= 4.0), Python (>= 3.9, TensorFlow 2.16+)

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

