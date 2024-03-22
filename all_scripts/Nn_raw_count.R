# Generate Nn read count table of a single experimental replicate.

# directory of the input pair-merged FASTQ files:
folder = '/rd1/home/liangn/Works/SEERS/N45_fastqs/L5/T2/R3'
# file names of the input pair-merged FASTQ files:
# nuclear DNA:
nDNA_fq = 'XH-ND_merged.fq.gz'
# nuclear RNA:
nRNA_fq = 'XH-NR_merged.fq.gz'
# cytoplasmic RNA:
cRNA_fq = 'XH-CR_merged.fq.gz'
# directory and name of the output file:
out = '/rd1/home/liangn/Works/SEERS/Nn_raw_counts/XH-R3_NnRawCount'

# flanking sequences of Nn:
forward_left_adapter = "GAGCTGTACAAGTAA"
forward_right_adapter = "TCTGTGCCTTCTAGT"

library(stringr)
library(data.table)
library(pbmcapply)
library(Biostrings)

# Generate str_match pattern for extracting Nn:
the_pattern <- paste(forward_left_adapter, "(.*?)", forward_right_adapter, sep='')
the_pattern
# Function to convert a fastq to Nn counts:
count.reads <- function(fq) {
  fq <- paste(folder, fq, sep='/')
  reads <- readDNAStringSet(fq, format = 'fastq', use.names = F)
  reads <- as.character(reads)
  Nn <- str_match(reads, the_pattern)
  Nn <- Nn[,2]
  Nn <- table(Nn)
  Nn <- as.data.table(Nn)
  names(Nn) <- c('Nn', 'count')
  return(Nn)
}
# Run counting:
counts.l <- pbmclapply(c(nDNA_fq, nRNA_fq, cRNA_fq), count.reads, mc.cores=3)
nDNA.dt <- counts.l[[1]]
nRNA.dt <- counts.l[[2]]
cRNA.dt <- counts.l[[3]]
nDNA.dt
nRNA.dt
cRNA.dt
rm(counts.l)
# Merge tables:
final_table <- merge(nDNA.dt, nRNA.dt, by='Nn', all=T, sort=F)
names(final_table) <- c('Nn', 'nDNA', 'nRNA')
final_table <- merge(final_table, cRNA.dt, by='Nn', all=T, sort=F)
names(final_table) <- c('Nn', 'nDNA', 'nRNA', 'cRNA')
final_table <- final_table[order(nDNA, decreasing=T)]
final_table
rm(nDNA.dt, nRNA.dt, cRNA.dt)
# Replace NA with 0:
final_table[is.na.data.frame(final_table)] <- 0
# Check N45 duplication:
which(duplicated(final_table$Nn))
# Check data:
final_table
str(final_table)
# Save output:
out_name <- paste(out, '.dt.rds', sep = '')
out_name
saveRDS(final_table, file = out_name)
