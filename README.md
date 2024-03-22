# SEERS (Selective Enrichment of Episomes containing Random Sequences)
Systematic exploration of 3'UTR regulatory elements and their contextual associations.
## Data Processing
Paired-end FASTQ files were first merged with NGmerge:
```sh
./NGmerge -d -1 1.fq.gz -2 2.fq.gz  -o merged.fq.gz
```
`Nn_raw_count.R` is used to count N45s from the FASTQ files.
`Nn_nclog2expression.R` is used to exclude noise from the N45 count results and infer their regulatory attributes.
## Model Training