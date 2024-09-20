# SEERS (Selective Enrichment of Episomes with Random Sequences)  
A systematic exploration of 3'UTR regulatory elements and their contextual associations.  

## Change Log
| Date         |  Description                                               |
| ------------ |  ------------------------------------------------------------ |
| 2024-08-14 | Resolved the issue preventing the model from loading after upgrading TensorFlow to 2.16. Refactored the Jupyter Notebook. |


## Data Processing
All paired-end FASTQ files were merged with `NGmerge`:
```sh
./NGmerge -d -1 1.fq.gz -2 2.fq.gz  -o merged.fq.gz
```
`Nn_raw_count.R` - Count N45s from the merged FASTQ files.  
`Nn_nclog2expression.R` - Exclude noise from the N45 count results, and infer their regulatory attributes.  
`ANN_data_prep.R` - Prepare data files for model training.  

## Model Training and Usage
`TALE_training_data.tar.bz2` - Training data.  
`TALE_train.ipynb` - Model training and evaluation.  
`L5-220528_em5-LSTM64x32x0.5-64x0.5-rep4.hdf5` - Our best "context-aware" model (TALE).  
`TALE_use.ipynb` - In silico experiments.  

## K-mer Analyses
`kmer_profiling.R` - Perform statistical tests for correlations between different k-mers and the regulatory phenotypes.  
`L5_2-8mer.tsv` - Our 2-8 mer profiling result from SEERS.  
`kmer_profile_visual.R` - Visualize the k-mer analysis result.  
