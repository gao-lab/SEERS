# SEERS: Selective Enrichment of Episomes with Random Sequences

**A systematic delineation of 3-prime UTR regulatory elements and their contextual associations.**

## Paper

This repository accompanies the SEERS study.

## Repository Contents

This repository is organized around two main components:

1. SEERS count/enrichment processing and k-mer analysis in `SEERS_data_processing/`
2. TALE model training, evaluation, and ClinVar ISM analysis in `TALE_model_260312/`

The materials in `TALE_models_260128_legacy/` are retained as a legacy reference.

## Folder Guide

### `SEERS_data_processing/`

- `Nn_pp.R`: extracts and counts N45 sequences from merged FASTQ files.
- `Nn_pp_pool.R`: pools sequence proportions across replicates.
- `combine_dna_cyt_nuc.R`: computes cytoplasmic and nuclear enrichment scores from DNA/Cyt/Nuc counts.
- `kmer_profiling.R`: performs k-mer association analyses.

These scripts use local input filenames configured near the top of each script. Run them from the directory containing the input files, or update `work_dir` and the input filenames explicitly.

### `TALE_model_260312/`

Recommended TALE model files.

- `model/model_bundle.pt`: selected pretrained PyTorch model bundle.
- `model/summary.json`: run configuration and metrics.
- `notebooks/TALE_training_260312.ipynb`: training notebook snapshot.
- `notebooks/TALE_ClinVar_ISM_v6.ipynb`: ClinVar SNV ISM logo notebook.
- `results/`: training histories and summary plots.
- `SHA256SUMS`: checksums for the bundled model and summary.

Selected run:

- `TALE_e5-lstm256bx128b-fc256-0.1-ema0.999-seed3407_260320_132204`
- Training data: `TALE_train_data_260312.csv`
- Independent A549 evaluation data: `3pL6-A549-T1.csv`
- Selected weights: EMA
- Best validation loss: 0.2024837457580668

### `TALE_models_260128_legacy/`

Legacy TALE scripts and pretrained weights retained for reference and comparison. For the current TALE model files, use `TALE_model_260312/`.

## Data

TALE training and evaluation data are distributed separately from this code repository.

For the current TALE workflow, use:

- `TALE_train_data_260312.csv`
- `3pL6-A549-T1.csv`

## Environment

Minimal requirements:

- R >= 4.0
- Python >= 3.9
- NGmerge for paired-end read merging

Python dependencies are listed in `requirements.txt`.

The optional Python package `umap-learn` is used only when running UMAP-based
legacy k-mer comparison code.

R package dependencies:

- Biostrings
- data.table
- ggplot2
- ggpointdensity
- pbmcapply
- scales
- viridis

## Quick Start

Prepare merged FASTQ files when starting from paired-end reads:

```bash
./NGmerge -d -1 read1.fq.gz -2 read2.fq.gz -o merged.fq.gz
```

Run SEERS count and enrichment processing:

```bash
cd SEERS_data_processing
Rscript Nn_pp.R
Rscript Nn_pp_pool.R
Rscript combine_dna_cyt_nuc.R
Rscript kmer_profiling.R
```

Run or inspect the updated TALE notebooks:

```bash
cd TALE_model_260312/notebooks
```

The training notebook expects `TALE_train_data_260312.csv` and `3pL6-A549-T1.csv` to be available locally. The ClinVar ISM notebook uses the bundled model by default and expects ClinVar input sequences to be supplied in the notebook parameter cell.
