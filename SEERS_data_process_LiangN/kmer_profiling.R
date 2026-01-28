#
work_dir = '/rd1/users/liangn/mywork/SEERS'
combined_data_file = '3pL5-HCT.dt.rds'
k = 8
out_prefix = '3pL5-HCT-8mer'
ncores = 8


library(data.table)
library(pbmcapply)
library(Biostrings)
#
setwd(work_dir)
#
combined_data <- readRDS(combined_data_file)
combined_data
# nuc and cyt scores:
nuc <- combined_data$nuc
nuc_scores <- log2(nuc+1)
rm(nuc)
cyt <- combined_data$cyt
cyt_scores <- log2(cyt+1)
rm(cyt)
# expression score:
expression_scores <- (nuc_scores+cyt_scores)/2
hist(expression_scores, breaks = 64)
# export score:
export_scores <- cyt_scores - nuc_scores
hist(export_scores, breaks = 64)
rm(nuc_scores, cyt_scores)
# form data.table
test.dt <- data.table(
  Nn               = combined_data$Nn,
  expression.score = expression_scores,
  export.score     = export_scores
)
test.dt
rm(combined_data, expression_scores, export_scores)
# global medians:
expression_score_median <- median(test.dt$expression.score, na.rm = TRUE)
expression_score_median
export_score_median <- median(test.dt$export.score, na.rm = TRUE)
export_score_median
# generate all k-mers:
N <- c("A", "T", "C", "G")
Grid     <- expand.grid(replicate(k, N, simplify = FALSE), stringsAsFactors = FALSE)
kmers.v  <- do.call(paste, c(Grid, list(sep = ""))) 
length(kmers.v)
4^k
kmers.v[1:16]

# make sure Nns are all upper case:
test.seq.upper <- toupper(as.character(test.dt$Nn))
test.seq.upper[1:32]

#
test_kmer <- function(i) {
  # define kmer:
  kmer <- toupper(kmers.v[i])
  # search Nns:
  hits_idx <- which(grepl(kmer, test.seq.upper, fixed = TRUE))
  # kmer rc:
  kmer_dna    <- DNAString(kmer)
  kmer_rc_dna <- reverseComplement(kmer_dna)
  kmer_rc     <- as.character(kmer_rc_dna)
  rc_idx <- which(grepl(kmer_rc, test.seq.upper, fixed = TRUE))
  # return NA if n <16:
  if (length(hits_idx) < 16L || length(rc_idx) < 16L) {
    return(NA)
  }
  #
  kmer.dt <- test.dt[hits_idx]
  rc.dt   <- test.dt[rc_idx]
  # kmer median scores:
  kmer_expression_score_median <- median(kmer.dt$expression.score, na.rm = TRUE)
  kmer_export_score_median     <- median(kmer.dt$export.score,     na.rm = TRUE)
  rc_expression_score_median   <- median(rc.dt$expression.score,   na.rm = TRUE)
  rc_export_score_median       <- median(rc.dt$export.score,       na.rm = TRUE)
  # delta scores:
  kmer_delta_expression.score  <- kmer_expression_score_median - expression_score_median
  kmer_delta_export.score      <- kmer_export_score_median     - export_score_median
  rc_delta_expression.score    <- rc_expression_score_median   - expression_score_median
  rc_delta_export.score        <- rc_export_score_median       - export_score_median
  # rc vs kmer:
  rc_expression.score_fc <- rc_delta_expression.score / kmer_delta_expression.score
  rc_export.score_fc     <- rc_delta_export.score     / kmer_delta_export.score
  # sd:
  kmer_expression.score_sd <- sd(kmer.dt$expression.score, na.rm = TRUE)
  kmer_export.score_sd     <- sd(kmer.dt$export.score,     na.rm = TRUE)
  rc_expression.score_sd   <- sd(rc.dt$expression.score,   na.rm = TRUE)
  rc_export.score_sd       <- sd(rc.dt$export.score,       na.rm = TRUE)
  # Mann–Whitney U test:
  expression_utest <- wilcox.test(
    kmer.dt$expression.score,
    test.dt$expression.score,
    exact = FALSE
  )
  export_utest <- wilcox.test(
    kmer.dt$export.score,
    test.dt$export.score,
    exact = FALSE
  )
  expression_pvalue <- expression_utest$p.value
  export_pvalue     <- export_utest$p.value
  # strand-specificity:
  pvalue_cutoff  <- 0.01
  rc_fc_cutoff   <- 2
  ss_expression  <- FALSE
  ss_export      <- FALSE
  ss_expression_utest <- wilcox.test(
    kmer.dt$expression.score,
    rc.dt$expression.score,
    exact = FALSE
  )
  ss_export_utest <- wilcox.test(
    kmer.dt$export.score,
    rc.dt$export.score,
    exact = FALSE
  )
  ss_expression_pvalue <- ss_expression_utest$p.value
  ss_export_pvalue     <- ss_export_utest$p.value
  if (ss_expression_pvalue < pvalue_cutoff &&
      (rc_expression.score_fc > rc_fc_cutoff ||
       rc_expression.score_fc < (1 / rc_fc_cutoff))) {
    ss_expression <- TRUE
  }
  if (ss_export_pvalue < pvalue_cutoff &&
      (rc_export.score_fc > rc_fc_cutoff ||
       rc_export.score_fc < (1 / rc_fc_cutoff))) {
    ss_export <- TRUE
  }
  # results:
  out_list <- list(
    kmer, length(hits_idx),
    kmer_delta_expression.score, expression_pvalue, kmer_expression.score_sd,
    kmer_delta_export.score,     export_pvalue,     kmer_export.score_sd,
    kmer_rc,
    ss_expression, rc_expression.score_fc, ss_expression_pvalue, rc_expression.score_sd,
    ss_export,     rc_export.score_fc,     ss_export_pvalue,     rc_export.score_sd
  )
  names(out_list) <- c(
    "kmer", "n",
    "delta.expression.score", "expression.p", "expression.score.sd",
    "delta.export.score",     "export.p",     "export.score.sd",
    "rc",
    "ss.expression", "rc.expression.score.fc", "ss.expression.p", "rc.expression.score.sd",
    "ss.export",     "rc.export.score.fc",     "ss.export.p",     "rc.export.score.sd"
  )
  return(out_list)
}

# test:
test_kmer(1)

# run:
kmer_test_results_raw <- pbmclapply(
  X = seq_along(kmers.v), 
  FUN = test_kmer, 
  mc.cores = ncores
)

# remove NA results (k-mers with insufficient n):
valid_idx <- vapply(
  kmer_test_results_raw,
  FUN = function(x) !(length(x) == 1 && all(is.na(x))),
  FUN.VALUE = logical(1)
)
valid_idx[1:8]
which(!valid_idx)
kmer_test_results <- kmer_test_results_raw[valid_idx]
# form data.table:
kmers.dt <- rbindlist(kmer_test_results, fill = TRUE)
# sort by kmer n:
setorder(kmers.dt, n)
kmers.dt

# save:
out_file <- paste0(out_prefix, ".dt.rds")
out_file
saveRDS(kmers.dt, out_file)
