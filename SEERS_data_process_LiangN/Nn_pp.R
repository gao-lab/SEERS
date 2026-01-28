#
work_dir = '/rd1/users/liangn/mywork/SEERS'
#
merged_fastq = 'merged.fq.gz'
#
left_adapter = 'GCATGGACGAGCTGTACAAGTAA'
#
right_adapter = 'TCTGTGCCTTCTAGTTGCCAG'


library(Biostrings)
library(data.table)
#
setwd(work_dir)
# read the FASTQ:
reads <- readDNAStringSet(merged_fastq, format = 'fastq')
reads
# G raw data:
length(reads)*300/1e9

# set adapters to trim:
left_adapter <- DNAString(left_adapter)
left_adapter
right_adapter <- DNAString(right_adapter)
right_adapter
#
left_adapter_length <- length(left_adapter)
left_adapter_length
right_adapter_length <- length(right_adapter)
right_adapter_length
adapter_total_length <- left_adapter_length + right_adapter_length
adapter_total_length
#
max_L_mismatch <- as.integer(round(left_adapter_length/20, digits = 0))
max_L_mismatch
max_R_mismatch <- as.integer(round(right_adapter_length/20, digits = 0))
max_R_mismatch
# trim adapters:
Nn <- trimLRPatterns(
  subject = reads, 
  Lpattern = left_adapter, 
  Rpattern = right_adapter,
  max.Lmismatch = max_L_mismatch, 
  max.Rmismatch = max_R_mismatch,
  with.Lindels = T, 
  with.Rindels = T
)
Nn
# remove non-trimmed:
trimmed <- width(reads) - width(Nn)
table(trimmed)
adapter_total_length
Nn <- Nn[trimmed == adapter_total_length]
Nn
# fraction of reads removed:
(length(reads)-length(Nn))/length(reads)
# check Nn sizes:
Nn_lengths <- width(Nn)
sample(Nn_lengths, 32)
hist(Nn_lengths, breaks = 1024)
# convert Nn to char:
Nn <- as.character(Nn)
Nn[1:8]
# count Nn and generate table:
Nn <- data.table(Nn = Nn)[, .(count = .N), by = Nn][order(-count)]
Nn
# check NA:
anyNA(Nn)
# check empty vector:
Nn[Nn=='']
# check Nn count distribution:
dt <- Nn
plot(1:nrow(dt), log2(dt$count), cex=0.2)
# zoom:
n=nrow(dt)/16
plot(1:nrow(dt[1:n]), log2(dt[1:n]$count), cex=0.2)
# convert to proportions:
total_counts <- sum(Nn$count)
total_counts
Nn$pp <- Nn$count / total_counts
Nn
Nn <- Nn[,c(1,3)]
Nn
sum(Nn$pp)
# save:
saveRDS(Nn, file='pp.dt.rds')
