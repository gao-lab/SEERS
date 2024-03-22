# An RDS or TSV table containing sequences and their log2 Cyt/DNA and Nuc/DNA values:
Nn_nclog2expr = 
  '/rd1/home/liangn/Works/SEERS-develop/all_experimental_data/SEERS_MPRA/experimental_Nn_log2expression/L5_220726_shuffled.dt.rds'
# Specify k-mer length:
k = 2
# Output directory and file name:
out = '/rd1/home/liangn/Works/SEERS-develop/L5-shuffled_2mer'
# Number of cores to use:
use_cores = 8

library(data.table)
library(pbmcapply)

# Function to read a table file:
load.table <- function(table_file) {
  require(data.table)
  table <- try(readRDS(table_file))
  if ("try-error" %in% class(table)) {
    table <- try(read.table(table_file, sep = '\t', header = T))
  }
  if ("try-error" %in% class(table)) {
    table <- read.csv(table_file, sep = '\t')
  }
  table <- as.data.table(table)
  return(table)
}
# Load the tables:
Nn.table <- load.table(Nn_nclog2expr)
Nn.table
str(Nn.table)
# Uniform data type:
wanted_cols <- c('Nn', 'nuc.log2expression', 'cyt.log2expression')
Nn.table <- Nn.table[, ..wanted_cols]
Nn.table$Nn <- as.character(Nn.table$Nn)
# Check duplication:
which(duplicated(Nn.table$Nn))
# Calculate log2(Cyt/Nuc):
Nn.table$log2export <- log2(2^Nn.table$cyt.log2expression/2^Nn.table$nuc.log2expression)
Nn.table
# Calculate log2(RNA/DNA):
Nn.table$log2expression <- log2(2^Nn.table$nuc.log2expression*(1/17)+2^Nn.table$cyt.log2expression*(16/17))
Nn.table
# Set table for test:
test.dt <- Nn.table[,c('Nn','log2expression', 'log2export')]
test.dt
str(test.dt)
rm(Nn.table)
#
hist(test.dt$log2expression, breaks = 128)
hist(test.dt$log2export, breaks = 128)
# Set whole-population medians:
log2expression_median <- median(test.dt$log2expression, na.rm=T)
log2export_median <- median(test.dt$log2export, na.rm=T)


# List all k-mers:
N <- c("A", "T", "C", "G")
Grid <- expand.grid(replicate(k, N, simplify = F))
kmers.v <- do.call(paste, c(Grid, list(sep='')))
length(kmers.v)
tail(kmers.v)

# function to test a k-mer:
library(Biostrings)
test.kmer <- function(kmer_i) {
  kmer <- kmers.v[kmer_i]
  # grep all Nn containing the k-mer:
  kmer_i <- grep(kmer, test.dt$Nn, ignore.case = T)
  kmer.dt <- test.dt[kmer_i]
  # standard deviation:
  kmer_log2expression_sd <- sd(kmer.dt$log2expression)
  kmer_log2export_sd <- sd(kmer.dt$log2export)
  # grep all Nn containing the reverse complement:
  kmer.dna <- DNAString(kmer)
  kmer_rc.dna <- reverseComplement(kmer.dna)
  kmer_rc <- as.character(kmer_rc.dna)
  rc_i <- grep(kmer_rc, test.dt$Nn, ignore.case = T)
  rc.dt <- test.dt[rc_i]
  # standard deviation:
  rc_log2expression_sd <- sd(rc.dt$log2expression)
  rc_log2export_sd <- sd(rc.dt$log2export)
  # Return NA if there were too few sequences to be tested:
  # This prevents error messages:
  if (nrow(kmer.dt)<16 | nrow(rc.dt)<16) {
    return(NA)
  } else {
    # number of useful Nn containing the k-mer:
    n <- nrow(kmer.dt)
    # k-mer expression median and delta:
    kmer_log2expression_median <- median(kmer.dt$log2expression, na.rm = T)
    kmer_delta_log2expression <- kmer_log2expression_median - log2expression_median
    # k-mer export median and delta:
    kmer_log2export_median <- median(kmer.dt$log2export, na.rm = T)
    kmer_delta_log2export <- kmer_log2export_median - log2export_median
    # Mann–Whitney U tests for p-values:
    expression_utest <- wilcox.test(kmer.dt$log2expression, test.dt$log2expression)
    expression_pvalue <- expression_utest$p.value
    export_utest <- wilcox.test(kmer.dt$log2export, test.dt$log2export)
    export_pvalue <- export_utest$p.value
    # reverse complement expression median, delta, and fold change:
    rc_log2expression_median <- median(rc.dt$log2expression, na.rm = T)
    rc_delta_log2expression <- rc_log2expression_median - log2expression_median
    rc_log2expression_fc <- rc_delta_log2expression / kmer_delta_log2expression
    # reverse complement export median, delta, and fold change:
    rc_log2export_median <- median(rc.dt$log2export, na.rm = T)
    rc_delta_log2export <- rc_log2export_median - log2export_median
    rc_log2export_fc <- rc_delta_log2export / kmer_delta_log2export
    # decide k-mer strand-specificity:
    ss_expression <- F
    ss_export <- F
    pvalue_cutoff = 0.01
    rc_fc_cutoff = 2
    ss_expression_utest <- wilcox.test(kmer.dt$log2expression, rc.dt$log2expression)
    ss_expression_pvalue <- ss_expression_utest$p.value
    ss_export_utest <- wilcox.test(kmer.dt$log2export, rc.dt$log2export)
    ss_export_pvalue <- ss_export_utest$p.value
    if (ss_expression_pvalue<pvalue_cutoff & 
        (rc_log2expression_fc>rc_fc_cutoff | rc_log2expression_fc<(1/rc_fc_cutoff))) {ss_expression <- T}
    if (ss_export_pvalue<pvalue_cutoff & 
        (rc_log2export_fc>rc_fc_cutoff | rc_log2export_fc<(1/rc_fc_cutoff))) {ss_export <- T}
    # organize results:
    return.ls <- list(
      kmer, n, 
      kmer_delta_log2expression, expression_pvalue, kmer_log2expression_sd,
      kmer_delta_log2export,     export_pvalue,     kmer_log2export_sd,
      kmer_rc,  
      ss_expression, rc_log2expression_fc, ss_expression_pvalue, rc_log2expression_sd,
      ss_export,     rc_log2export_fc,     ss_export_pvalue,     rc_log2export_sd
    )
    names(return.ls) <- c(
      'kmer', 'n', 
      'delta.log2expression', 'expression.p', 'log2expression.sd',
      'delta.log2export',     'export.p',     'log2export.sd',
      'rc', 
      'ss.expression', 'rc.log2expression.fc', 'ss.expression.p', 'rc.log2expression.sd',
      'ss.export',     'rc.log2export.fc',     'ss.export.p',     'rc.log2export.sd'
    )
    return(return.ls)
  }
}
test.kmer(1)
# analyze all k-mers:
kmer_test_results <- pbmclapply(1:length(kmers.v), test.kmer, mc.cores = use_cores)
which(is.na(kmer_test_results))
kmer_test_results <- kmer_test_results[!is.na(kmer_test_results)]
# make result table:
kmers.dt <- rbindlist(kmer_test_results)
kmers.dt <- kmers.dt[order(kmers.dt$delta.log2export, decreasing = F)]
kmers.dt

# Save:
out.tsv <- paste(out, '.tsv', sep = '')
out.tsv
write.table(kmers.dt, out.tsv, quote = F, sep = '\t', row.names = F)
