#
work_dir = 
  '/rd1/home/liangn/Works/SEERS-develop/kmers/kmer_profiles'
#
kmer_profile = 
  'L5_2-8mer.tsv'

library(data.table)
library(ggplot2)
library(Biostrings)
library(pbmcapply)

#
setwd(work_dir)
# load data:
kmer_profile <- read.table(kmer_profile, header = T, sep = '\t')
kmer_profile <- as.data.table(kmer_profile)
kmer_profile$kmer <- as.character(kmer_profile$kmer)
kmer_profile$rc <- as.character(kmer_profile$rc)
kmer_profile
str(kmer_profile)
#
kmer_profile$ss <- kmer_profile$ss.expression | kmer_profile$ss.export
kmer_profile$k <- nchar(kmer_profile$kmer)

#
kmer_profile[order(delta.log2expression, decreasing = T)][1:16]

# expression sd:
hist(kmer_profile$log2expression.sd, breaks = 128)
median(kmer_profile$log2expression.sd)
kmer_profile[log2expression.sd>1.49][log2expression.sd<1.57][order(delta.log2expression, decreasing = T)][1:16]
# sd vs. expression:
ggplot(kmer_profile, aes(x=log2expression.sd, y=delta.log2expression)) + 
  geom_point(size=0.2) +
  theme_bw()+
  theme(aspect.ratio=1/1)+
  xlab('SD') +
  ylab('Delta log2(RNA/DNA)') + 
  theme(axis.text = element_text(size=11),
        axis.title = element_text(size=13),
        legend.title = element_text(size=13),
        legend.text = element_text(size=11)) +
  geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.2, alpha = 1) +
  geom_vline(xintercept = 1.532407, linetype = "solid", color = "black", linewidth=0.2, alpha = 1)
#
hist(kmer_profile$log2export.sd, breaks = 128)

# k-mer lookup:
kmer_profile[kmer=='CCGGC']


# k-mer 2D with label:
dt <- kmer_profile
#dt <- dt[delta.log2export>-0.3][delta.log2expression>-0.5][export.p<1e-6|expression.p<1e-6]
ggplot(dt, aes(x=delta.log2export, y=delta.log2expression, color=ss)) + 
  xlim(-0.005, 0.025) + ylim(-0.04, 0.04) +
  geom_point(size=0.4) +
  theme_bw()+
  theme(aspect.ratio=1/1)+
  scale_colour_manual(name='Strand specific', values=setNames(c('blue','red'), c(F, T))) +
  xlab('Delta log2(Cyt/Nuc)') + ylab('Delta log2(RNA/DNA)') +
  theme(axis.text = element_text(size=13),
        axis.title = element_text(size=15),
        legend.title = element_text(size=15),
        legend.text = element_text(size=13)) +
  geom_hline(yintercept=0, linetype="solid", color="black", linewidth=0.2, alpha=1) +
  geom_vline(xintercept=0, linetype="solid", color="black", linewidth=0.2, alpha=1) +
  geom_text(label=dt$kmer, color='black', size=2.7, 
            vjust="inward", hjust="outward",
            nudge_x=0, nudge_y=0, check_overlap=T)


# k-mer expression U-test result:
ggplot(kmer_profile, aes(x=delta.log2expression, y=-log10(expression.p), color=ss.expression)) + 
  xlim(-0.04, 0.04) + ylim(0, 150) +
  geom_point(size=0.4) +
  theme_bw()+
  theme(aspect.ratio=1/1) +
  scale_colour_manual(name='Strand specific', values=setNames(c('blue','red'), c(F, T))) +
  xlab('Delta log2(RNA/DNA)') + ylab('-log10(p-value)') +
  theme(axis.text = element_text(size=13),
        axis.title = element_text(size=15),
        legend.title = element_text(size=15),
        legend.text = element_text(size=13)) +
  geom_hline(yintercept = 0, linetype = "solid", color = "black", size = 0.2, alpha = 1) +
  geom_vline(xintercept = 0, linetype = "solid", color = "black", size=0.2, alpha = 1) +
  geom_hline(yintercept = 5, linetype = "dashed", color = "black", size = 0.2, alpha = 1) +
  geom_text(label=kmer_profile$kmer, color='black', size=2.7, 
            vjust="inward", hjust="outward",
            nudge_x=0, nudge_y=0, check_overlap=T)

# k-mer localization U-test result:
ggplot(kmer_profile, aes(x=delta.log2export, y=-log10(export.p), color=ss.export)) + 
  xlim(-0.005, 0.025) + ylim(0, 120) +
  geom_point(size=0.4) +
  theme_bw() +
  theme(aspect.ratio=1/1) +
  scale_colour_manual(name='Strand specific', values=setNames(c('blue','red'), c(F, T))) +
  xlab('Delta log2(Cyt/Nuc)') + ylab('-log10(p-value)') +
  theme(axis.text = element_text(size=13),
        axis.title = element_text(size=15),
        legend.title = element_text(size=15),
        legend.text = element_text(size=13)) +
  geom_hline(yintercept = 0, linetype = "solid", color = "black", size = 0.2, alpha = 1) +
  geom_vline(xintercept = 0, linetype = "solid", color = "black", size=0.2, alpha = 1) +
  geom_hline(yintercept = 5, linetype = "dashed", color = "black", size = 0.2, alpha = 1) +
  geom_text(label=kmer_profile$kmer, color='black', size=2.7, 
            vjust="inward", hjust="outward",
            nudge_x=0, nudge_y=0, check_overlap=T)

# cytoplasmic k-mers:
dt <- kmer_profile[order(delta.log2export, decreasing = T)]
dt[, c('kmer','delta.log2export','export.p','n')][1:16]
# high-expression k-mers:
dt <- kmer_profile[order(delta.log2expression, decreasing = T)]
dt[1:8]
# nuclear k-mers:
dt <- kmer_profile[order(delta.log2export, decreasing = F)][delta.log2expression>=0]
dt[1:8]
# low-expression k-mers:
dt <- kmer_profile[order(delta.log2expression, decreasing = F)]
dt[1:8]


# exam n:
kmer_profile[order(n)]
hist(kmer_profile$n, breaks = 128)
barplot(
  # xlim = c(0,100), ylim = c(0,500),
  height=sort(kmer_profile$n), ylab='n', xlab='kmer')


#
dt <- kmer_profile[cluster=='HC3']
dt

# SD vs. k:
dt <- kmer_profile[expression.p<1e-6]
ggplot(dt, aes(x=k, y=log2expression.sd)) + 
  geom_point(size=0.1) +
  theme_bw()+
  theme(aspect.ratio=1/1)+
  xlab('k') + ylab('SD') +
  theme(axis.text = element_text(size=11),
        axis.title = element_text(size=13),
        legend.title = element_text(size=13),
        legend.text = element_text(size=11)) 

# density:
ggplot(kmer_profile, aes(x=delta.log2expression, y=log2expression.sd)) +
  geom_bin2d(bins = 128, aes(fill = ..count..)) +
  scale_fill_continuous(type = "viridis", name = 'Count') +
  theme_bw()+
  xlab('Delta log2(RNA/DNA)') + ylab('SD') +
  theme(aspect.ratio=1/1,
        axis.text = element_text(size=11),
        axis.title = element_text(size=13),
        legend.title = element_text(size=13),
        legend.text = element_text(size=11)) +
  geom_hline(yintercept = 0, color = "red", size = 0.5, alpha = 1) +
  geom_vline(xintercept = 0, color = "red", size=0.5, alpha = 1) 
#
kmer_profile[expression.p<1e-6 | export.p<1e-6]
18925/87376

# k-mer 2D density:
dt <- kmer_profile
dt <- dt[delta.log2export>-0.3][delta.log2expression>-0.5]
dt <- dt[export.p<1e-5 | expression.p<1e-5]
dt <- dt[k==8]
dt
ggplot(dt, aes(x=delta.log2export, y=delta.log2expression)) +
  geom_bin2d(bins = 32, aes(fill = ..count..)) +
  scale_fill_continuous(type = "viridis", name = 'Count') +
  theme_bw()+
  xlab('Delta log2(Cyt/Nuc)') + ylab('Delta log2(RNA/DNA)') +
  theme(aspect.ratio=1/1,
        axis.text = element_text(size=11),
        axis.title = element_text(size=13),
        legend.title = element_text(size=13),
        legend.text = element_text(size=11)) +
  geom_hline(yintercept = 0, color = "red", size = 0.5, alpha = 1) +
  geom_vline(xintercept = 0, color = "red", size=0.5, alpha = 1) +
  xlim(-0.3, 0.25)




# clusters:
nlevels(kmer_profile$cluster)
n_members <- table(kmer_profile$cluster)
n_members <- sort(n_members, decreasing = T)
n_members[1]
barplot(n_members, xlim = c(0,100))
length(which(n_members>50))
