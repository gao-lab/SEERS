This script trains a PyTorch LSTM-based sequence model to predict nuclear and cytoplasmic signals in A549 and HCT116 cells from DNA sequences.

Prepare train_set.csv, val_set.csv, and test_set.csv under train_data_260106/, each containing a sequence column (Nn) and four target columns:
nuc.score.A549, cyt.score.A549, nuc.score.HCT116, cyt.score.HCT116.

Run training and evaluation with:
python Re_trained_seerr.py




This script sweeps Conv1D kernel sizes for a simple CNN (Conv1D + GlobalMaxPool + MLP) to predict four targets (A549 nuc/cyt, HCT116 nuc/cyt). It trains across multiple random seeds, saves per-(seed,kernel) checkpoints, and outputs per-seed + aggregated (mean/std) CSVs and error-bar plots.
python eval_cnn1_kernel_sweep.py \
  --data_dir ./train_data_260106 \
  --out_dir ./cnn1_kernel_sweep_out_seeds \
  --seeds 42 0 5314 \
  --kernel_min 2 --kernel_max 11 \


This script evaluates CNN1 kernel-sweep checkpoints on an external A549 test set.
It loads all seed*/ks*.pt checkpoints, keeps the first two outputs ([A549_nuc, A549_cyt]), and reports MAE / Pearson / R².

python eval_external_models_on_3pL6_A549.py \
  --models_dir ./cnn1_kernel_sweep_out_seeds/models \
  --test_csv ./train_data_260106/3pL6-A549.csv \
  --out_csv ./cnn1_kernel_sweep_out_seeds/external_test_3pL6-A549.metrics.csv \
  --gpu 0

This script evaluates trained SEERS LSTM models on an external A549 dataset.
It loads final_model.pth (or best_model.pth) from each seed* directory, runs inference, and reports MAE / Pearson / R² for A549 nuclear and cytoplasmic outputs.

python eval_seers_lstm_on_3pL6_A549.py \
  --models_root ./models/ \
  --which best \
  --test_csv ./train_data_260106/3pL6-A549.csv \
  --out_csv ./models//external_test_3pL6-A549.lstm_metrics.csv \
  --gpu 0

This script evaluates a single trained SEERS LSTM model on an external HCT116 dataset.
It loads one final_model.pth, extracts the HCT116 outputs ([HCT116_nuc, HCT116_cyt]), computes MAE / Pearson / R², and reports prefix-based R² (top-K samples). It also saves scatter plots (observed vs predicted).

python external_test_3pL6_HCT116.py \
  --model_path ./models/final_model.pth \
  --test_csv ./train_data_260106/3pL4-HCT116-T1.csv \
  --gpu 0 \
  --prefix_list "[1000,2000,5000,10000,20000]"

This script analyzes k-mer distribution overlap between two sequence sets (e.g. human 3′UTRs vs random/control sequences).
It computes k-mer frequency vectors, embeds them into a shared 2D space (UMAP or PCA), and saves co-embedding coordinates for downstream visualization or analysis.

python analyze_kmer_overlap.py



This script computes SNP effects in ClinVar 3′UTRs using a PyTorch sequence model.
Each 89-nt ref/alt sequence is split into overlapping 45-mers, encoded with the same tokens as the original Keras model, and evaluated in batch.
The SNP effect is reported as the median Δlog2(Cyt/DNA) across all windows.

python eval_clinvar_snps_pytorch.py \
  --model_path /rd4/users/liangn/your_pytorch_model.pth \
  --clinvar_tsv /rd4/users/liangn/ClinVar_3UTR_SNPs.tsv \
  --gpu 7 \
  --do_random --n_random 10000 \
  --random_hist_png random_snp_effects.pytorch.png

