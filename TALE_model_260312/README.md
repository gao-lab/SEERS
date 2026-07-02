# TALE Model

This folder contains the updated TALE PyTorch model and analysis notebooks.

## Contents

- `model/model_bundle.pt`: selected pretrained model bundle from the 2026-03-20 run.
- `model/summary.json`: model, preprocessing, training, and evaluation summary.
- `notebooks/TALE_training_260312.ipynb`: training notebook snapshot for the selected run.
- `notebooks/TALE_ClinVar_ISM_v6.ipynb`: ClinVar SNV ISM logo notebook.
- `results/`: lightweight training/evaluation histories and summary figures.
- `SHA256SUMS`: checksums for the bundled model and summary.

The source training table `TALE_train_data_260312.csv` and independent A549 table `3pL6-A549-T1.csv` are distributed separately from this code repository.

## Selected Run

- Run name: `TALE_e5-lstm256bx128b-fc256-0.1-ema0.999-seed3407_260320_132204`
- Selected weights: EMA
- Seed: 3407
- Best validation loss: 0.2024837457580668
- Training data: `TALE_train_data_260312.csv`
- Independent A549 evaluation data: `3pL6-A549-T1.csv`

Key metrics are available in `model/summary.json`.

## Notebook Use

Run notebooks from this directory layout so relative paths resolve as expected:

```bash
cd TALE_model_260312/notebooks
```

For the ClinVar ISM logo notebook, edit the final parameter cell to set `REF89`, `ALT_BASE`, and `TARGET_NAME`.
