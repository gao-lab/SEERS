# Filter N45 and calculate their average expressions in nuclear and 
# cytoplasmic contents based on multiple experimental replicates.

# directory of the Nn read count tables (RDS files) from previous step:
folder = '/rd1/home/liangn/Works/SEERS-develop/experiment_Nn_raw_counts/L5/tech_rep2'
# the expected Nn length:
Nn_length = 45
# the +- allowance of the Nn length:
Nn_length_allow = 1
# output directory and file name:
out = '/rd1/home/liangn/Works/SEERS-develop/L5_tech_rep2_nclog2expression'


library(data.table)

# get all file paths:
file_paths <- list.files(path = folder, full.names = T, ignore.case = T)
file_paths
# load data tables:
all_files <- lapply(file_paths, readRDS)
all_files
str(all_files)
# skip merging if only had one file:
if (length(all_files)==1) {
  merged_counts <- all_files[[1]]
}
# function to merge multiple tables:
merge.tables <- function(tablelist, byname, all) {
  library(stringr)
  col_names.v <- colnames(tablelist[[1]])
  col_names.v <- col_names.v[which(col_names.v!=byname)]
  merged.dt <- merge(tablelist[[1]], tablelist[[2]], by=byname, all=all, sort=F)
  colnames(merged.dt) <- str_replace(names(merged.dt), '.x$', '.1')
  colnames(merged.dt) <- str_replace(names(merged.dt), '.y$', '.2')
  if (length(tablelist)>2) {
    for (i in 3:length(tablelist)) {
      merged.dt <- merge(merged.dt, tablelist[[i]], by=byname, all=all, sort=F)
      for (thename in col_names.v) {
        names(merged.dt)[names(merged.dt)==thename] <- paste(thename,i,sep='.')
      }
    }
  }
  return(merged.dt)
}
# Merge the count tables:
if (length(all_files)>1) {
  merged_counts <- merge.tables(all_files, 'Nn', all = T)
}
merged_counts
# replace NA by 0:
merged_counts[is.na.data.frame(merged_counts)] <- 0
merged_counts
# function to grep table columns by names:
grep.cols <- function(col_name, dt) {
  library(data.table)
  col_i <- grep(col_name, names(dt), ignore.case=T)
  sub_table <- dt[, ..col_i]
  return(sub_table)
}

# filter Nn by DNA:
DNA.dt <- grep.cols('DNA', merged_counts)
DNA.dt
# check occurrence in replicates:
Nn_reps <- rowSums(DNA.dt>0)
hist(Nn_reps, breaks = 64)
# set replicate cutoff:
min_rep = 2
# check number of reads:
Nn_DNA_reads <- rowSums(DNA.dt)
hist(Nn_DNA_reads, breaks = 256)
hist(log2(Nn_DNA_reads), breaks = 256)
# set reads cutoff:
min_count = 2^2.5
# perform 1st filtration:
filtered_Nn_counts <- merged_counts[Nn_DNA_reads>=min_count & Nn_reps>=min_rep]
filtered_Nn_counts
# 2nd filtration:
DNA.dt <- grep.cols('DNA', filtered_Nn_counts)
DNA.dt
# check occurrence:
Nn_reps <- rowSums(DNA.dt>0)
hist(Nn_reps, breaks = 64)
# reset replicate cutoff:
min_rep = 2
# check number of reads:
Nn_DNA_reads <- rowSums(DNA.dt)
hist(Nn_DNA_reads, breaks = 256)
hist(log2(Nn_DNA_reads), breaks = 128)
# reset reads cutoff:
min_count = 2^(2.5)
min_count
# perform 2nd filtration:
filtered_Nn_counts <- filtered_Nn_counts[Nn_DNA_reads>=min_count & Nn_reps>=min_rep]
filtered_Nn_counts
# nRNA>0:
nRNA.dt <- grep.cols('nRNA', filtered_Nn_counts)
nRNA.dt
filtered_Nn_counts <- filtered_Nn_counts[rowSums(nRNA.dt)>0]
# cRNA>0:
cRNA.dt <- grep.cols('cRNA', filtered_Nn_counts)
cRNA.dt
filtered_Nn_counts <- filtered_Nn_counts[rowSums(cRNA.dt)>0]
# check:
filtered_Nn_counts
# Filter Nn by length:
Nn_lengths <- nchar(filtered_Nn_counts$Nn)
hist(Nn_lengths, breaks = 2^9)
hist(Nn_lengths, breaks = 2^8, xlim = c(Nn_length-10, Nn_length+10))
good_lengths <- (Nn_lengths>=(Nn_length-Nn_length_allow))&(Nn_lengths<=(Nn_length+Nn_length_allow))
filtered_Nn_counts <- filtered_Nn_counts[good_lengths]
filtered_Nn_counts
table(nchar(filtered_Nn_counts$Nn))

# check Nn reads coverages:
hist(log2(rowSums(grep.cols('nDNA', filtered_Nn_counts))), breaks = 64)
hist(log2(rowSums(grep.cols('nRNA', filtered_Nn_counts))), breaks = 64)
hist(log2(rowSums(grep.cols('cRNA', filtered_Nn_counts))), breaks = 64)


# calculate Nn reads proportions:
prop.dt <- filtered_Nn_counts[, -c('Nn')]
prop.dt
for (i in 1:ncol(prop.dt)) {
  prop.dt[[i]] <- prop.dt[[i]] / sum(prop.dt[[i]])
}
prop.dt
colSums(prop.dt)

# check sample correlations:
prop.dt$NC <- rnorm(nrow(prop.dt), sd=sd(prop.dt[[1]], na.rm=T))
cor.mt <- cor(prop.dt)
heatmap(abs(cor.mt), margins=c(8,8), col=colorRampPalette(c("white","red3","black"))(256),
        scale='none', cexRow=1, cexCol=1)

# pool the proportions:
filtered_Nn_counts$nDNA <- rowMeans(grep.cols('nDNA', prop.dt), na.rm = T)
filtered_Nn_counts$nRNA <- rowMeans(grep.cols('nRNA', prop.dt), na.rm = T)
filtered_Nn_counts$cRNA <- rowMeans(grep.cols('cRNA', prop.dt), na.rm = T)
filtered_Nn_counts
colSums(filtered_Nn_counts[,-c('Nn')])
# check no 0 proportions:
min(filtered_Nn_counts$nDNA)
min(filtered_Nn_counts$nRNA)
min(filtered_Nn_counts$cRNA)

# calculate nuclear log2expression:
filtered_Nn_counts$nuc.log2expression <- log2(filtered_Nn_counts$nRNA/filtered_Nn_counts$nDNA)
hist(filtered_Nn_counts$nuc.log2expression, breaks = 2^6)
# calculate cytoplasmic expression:
filtered_Nn_counts$cyt.log2expression <- log2(filtered_Nn_counts$cRNA/filtered_Nn_counts$nDNA)
hist(filtered_Nn_counts$cyt.log2expression, breaks = 2^6)
# order Nn by DNA:
filtered_Nn_counts <- filtered_Nn_counts[order(nDNA, decreasing = T)]
filtered_Nn_counts


# save:
out_name <- paste(out, '.dt.rds', sep = '')
out_name
saveRDS(filtered_Nn_counts, out_name)
