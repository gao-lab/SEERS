## SEERS: Selective Enrichment of Episomes with Random Sequences

**A systematic delineation of 3′ UTR regulatory elements and their contextual associations.**

---

### 📄 Paper

This repository accompanies the SEERS manuscript (BioRxiv):  
https://www.biorxiv.org/content/10.1101/2025.06.09.658412v2

---

### 📌 What this repository currently contains

This repo is focused on two practical parts:

1. **SEERS count/enrichment processing and k-mer analysis (R scripts)**  
   Folder: `SEERS_data_process_LiangN/`
2. **TALE model training/evaluation scripts and a pretrained weight (Python scripts)**  
   Folder: `TALE_models_LiJY_260128/`

---

### 📂 Current folder guide (based on existing files)

#### `SEERS_data_process_LiangN/` (R)

- `Nn_pp.R`  
  Extracts and counts N45 sequences from merged FASTQ files.
- `Nn_pp_pool.R`  
  Pools counts across biological/technical replicates.
- `combine_dna_cyt_nuc.R`  
  Computes enrichment-style values using DNA/Cytoplasm/Nucleus counts.
- `kmer_profiling.R`  
  Performs k-mer level association/correlation analyses.

#### `TALE_models_LiJY_260128/` (Python)

- `Re_trained_seerr.py`  
  Main retraining script for SEERS/TALE style modeling.
- `eval_seers_lstm_on_3pL6_A549.py`  
  Evaluation script for LSTM model on A549-related set.
- `external_test_3pL6_HCT116.py`  
  External test script for HCT116-related set.
- `eval_external_models_on_3pL6_A549.py`  
  Compare/evaluate external models on A549-related set.
- `eval_clinvar_snps_pytorch.py`  
  ClinVar SNP effect evaluation utility.
- `eval_cnn1_kernel_sweep.py`  
  CNN kernel sweep evaluation script.
- `Randomseq_vs3utr.py`  
  Utility for random-sequence vs 3′UTR comparisons.
- `model/final_model.pth`  
  Included pretrained model weight.

---

### 🚀 Quick start (current scripts)

### 1) Prepare merged FASTQ (if starting from paired-end reads)

Use NGmerge as in the manuscript workflow:

```bash
./NGmerge -d -1 read1.fq.gz -2 read2.fq.gz -o merged.fq.gz
```

### 2) Run R-based processing

```bash
cd SEERS_data_process_LiangN
Rscript Nn_pp.R
Rscript Nn_pp_pool.R
Rscript combine_dna_cyt_nuc.R
Rscript kmer_profiling.R
```

> Note: these scripts assume your input paths/filenames are configured inside scripts.

### 3) Run Python model scripts

```bash
cd TALE_models_LiJY_260128
python Re_trained_seerr.py
python eval_seers_lstm_on_3pL6_A549.py
python external_test_3pL6_HCT116.py
python eval_external_models_on_3pL6_A549.py
python eval_clinvar_snps_pytorch.py
python eval_cnn1_kernel_sweep.py
```

---

### 📦 Data

Full training dataset (Zenodo):  
https://doi.org/10.5281/zenodo.18737939

---

### 🧰 Environment (minimal)

- R >= 4.0
- Python >= 3.9
- NGmerge

If you encounter package errors, please install dependencies required by each script in your local environment.

---

### 📝 Changelog

| Date | Version/Update | Description |
| --- | --- | --- |
| **2026-01-28** | v2.0 | Updated scripts for revised manuscript and new datasets. |
| **2025-04-21** | v1.2 | Added `kmer_motif.ipynb` and `N45_dissect.ipynb`. |
| **2024-12-08** | v1.1 | Added `TALE_SNP_effect.ipynb`. |
| **2024-08-14** | v1.0 | TF 2.16 compatibility fix & Refactored Notebooks. |

