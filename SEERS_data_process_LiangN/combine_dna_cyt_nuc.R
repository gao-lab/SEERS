#
work_dir = '/rd1/users/liangn/mywork/SEERS'
#
dna = 'DNA-T1.pp.dt.rds'
cyt = 'Cyt-T1.pp.dt.rds'
nuc = 'Nuc-T1.pp.dt.rds'
#
out = '3pL5-A549-T1'


library(data.table)
# load:
setwd(work_dir)
dna <- readRDS(dna)
dna
cyt <- readRDS(cyt)
cyt
nuc <- readRDS(nuc)
nuc

# merge:
merged <- merge(dna, cyt, by='Nn', all.x = T, sort=F, suffixes = c('.dna','.cyt'))
merged
rm(dna, cyt)
merged[is.na(merged)] <- 0
merged
merged <- merge(merged, nuc, by='Nn', all.x = T, sort=F)
merged
rm(nuc)
colnames(merged)[4] <- 'pp.nuc'
merged
merged[is.na(merged)] <- 0
merged
# sort:
merged <- merged[order(pp.dna, decreasing = T)]
merged

# size select:
widths <- nchar(merged$Nn)
widths[1:16]
hist(widths, breaks = 1024)
hist(widths, breaks = 1024, xlim=c(40,50))
merged <- merged[widths>=44 & widths<=46]
merged

# remove non-reliable Nn:
plot(1:nrow(merged), log2(merged$pp.dna), cex=0.2)
# set cutoff at cliff:
top_n = 8e4
merged <- merged[1:top_n]
merged

# re-normalize:
sum(merged$pp.dna)
merged$pp.dna <- merged$pp.dna / sum(merged$pp.dna)
sum(merged$pp.dna)
sum(merged$pp.cyt)
merged$pp.cyt <- merged$pp.cyt / sum(merged$pp.cyt)
sum(merged$pp.cyt)
sum(merged$pp.nuc)
merged$pp.nuc <- merged$pp.nuc / sum(merged$pp.nuc)
sum(merged$pp.nuc)

# function to plot scatter with density colors:
plot_scatter <- function(x, y, size=1, adjust=1, neutral=1) {
  # form table:
  dt <- data.table()
  dt$x <- x
  dt$y <- y
  # remove rows containing Inf:
  dt <- dt[is.finite(x) & is.finite(y)]
  # set plot range:
  x_range <- range(dt$x)
  y_range <- range(dt$y)
  common_range <- range(x_range, y_range)
  common_range
  # scatter plot with density:
  library(ggplot2)
  library(ggpointdensity)
  library(viridis)
  library(scales)
  p <- ggplot(dt, aes(x, y)) +
    geom_pointdensity(aes(colour = after_stat(log2(density))),
                      size = size, adjust = adjust) +
    scale_colour_viridis_c(
      option = "turbo",
      name   = expression(log[2]*"(density)"),
      labels = label_number(accuracy = 1)
    ) +
    scale_x_continuous(
      limits = common_range,
      breaks = function(lim) unique(round(pretty(lim))),
      labels = scales::label_number(accuracy = 1)
    ) +
    scale_y_continuous(
      limits = common_range,
      breaks = function(lim) unique(round(pretty(lim))),
      labels = scales::label_number(accuracy = 1)
    ) +
    coord_fixed(ratio = 1) +
    theme_bw(base_size = 17, base_family = "Helvetica") +
    xlab("x") + ylab("y") +
    theme(aspect.ratio = 1) +
    geom_hline(yintercept = neutral, colour = "black", linewidth = 0.5) +
    geom_vline(xintercept = neutral, colour = "black", linewidth = 0.5) +
    geom_abline(intercept = 0, slope = 1,
                colour = "red", linetype = "dashed", linewidth = 0.5)
  #
  return(p)
}

# compare cyt nuc fc:
s = 0.01
x <- log2(merged$pp.nuc / merged$pp.dna +s)
y <- log2(merged$pp.cyt / merged$pp.dna +s)
# R2:
r <- cor(x, y)
r
rho <- cor(x, y, method = 'spearman')
rho
# plot:
p <- plot_scatter(x, y, size=0.5, adjust=2, neutral=log2(1 +s))
p

# add cyt nuc scores:
merged
merged$cyt.score <- y
merged$nuc.score <- x
merged
merged <- merged[,c('Nn','pp.dna','cyt.score','nuc.score')]
merged
colnames(merged)[2] <- 'dna'
merged

# save:
out
out_file <- paste(out, '.dt.rds', sep='')
out_file
saveRDS(merged, out_file)
# save .csv:
out
out_file <- paste(out, '.csv', sep='')
out_file
fwrite(merged, out_file)
