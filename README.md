# SEERS (Selective Enrichment of Episomes containing Random Sequences)
Systematic exploration of 3'UTR regulatory elements and their contextual associations.  

## Data Processing
All paired-end FASTQ files were merged with `NGmerge`:
```sh
./NGmerge -d -1 1.fq.gz -2 2.fq.gz  -o merged.fq.gz
```
`Nn_raw_count.R` - Count N45s from the merged FASTQ files.  
`Nn_nclog2expression.R` - Exclude noise from the N45 count results, and infer their regulatory attributes.  
`ANN_data_prep.R` - Prepare data files for model training.  

## Model Training and Usage
`SEERS_train.ipynb` - Model training and evaluation.  
`SEERS_evolution.ipynb` - In silico experiments.  

## K-mer Analyses
`kmer_profiling.R` - Perform statistical tests for correlations between different k-mers and the regulatory phenotypes.  
`kmer_profile_visual.R` - Visualize the k-mer analysis result.  