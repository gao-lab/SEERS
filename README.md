# SEERS (Selective Enrichment of Episomes with Random Sequences)  
A systematic delineation of 3′ UTR regulatory elements and their contextual associations.

## Change Log
| Date         |  Description                                               |
| ------------ |  ------------------------------------------------------------ |
| 2024-08-14 | Resolved the issue preventing the model from loading after upgrading TensorFlow to 2.16. Refactored the Jupyter Notebook. |
| 2024-12-08 | Added `TALE_SNP_effect.ipynb` |
| 2025-04-21 | Added `kmer_motif.ipynb` and `N45_dissect.ipynb` |
| 2026-01-28 | Updated all scripts to reflect the revised manuscript and updated dataset. |

## SEERS Data Processing & k-mer Analyses
`SEERS_data_process_LiangN_260128` - Refer to this folder for now.

All paired-end FASTQ files were merged with `NGmerge`:
```sh
./NGmerge -d -1 1.fq.gz -2 2.fq.gz  -o merged.fq.gz
```

`Nn_pp.R` - Count N45 from `merged.fq.gz` files.  
`Nn_pp_pool.R` - Pool N45 counts.  
`combine_dna_cyt_nuc.R` - Generate SEERS data from counts.  
`kmer_profiling.R` - Test k-mers for their regulatory correlations.

## Model Training and Usage
`TALE_models_LiJY_260128` - Refer to this folder for now.