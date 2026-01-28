#
work_dir = '/rd1/users/liangn/mywork/SEERS'
sample1 = 
  's1.pp.dt.rds'
sample2 = 
  's2.pp.dt.rds'
weight1 = 1
weight2 = 1
out = 
  's1s2'
plot_compare = T


library(data.table)
# load:
setwd(work_dir)
sample1 <- readRDS(sample1)
sample1
sample2 <- readRDS(sample2)
sample2
# merge:
merged <- merge(sample1, sample2, by='Nn', all=T, sort=F, suffixes = c('.sample1','.sample2'))
merged
# replace NA with 0:
merged[is.na(merged)] <- 0
merged

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
# remove rows with 0:
dt <- merged[,c(2,3)]
dt
non_zero_rows <- rowSums(dt==0)==0
non_zero_rows[1:8]
dt <- merged[non_zero_rows]
dt
x <- log10(dt$pp.sample1)
y <- log10(dt$pp.sample2)
# R2:
r <- cor(x, y)
r
rho <- cor(x, y, method = 'spearman')
rho
# plot:
if (plot_compare == T) {
  p <- plot_scatter(x, y, size=0.5, adjust=1, neutral=0)
  p
}

# pool Nn proportions by weights:
dt <- merged[, -1]
dt
anyNA(dt)
colSums(dt)
#
pooled_pp <- dt[[1]]*weight1 + dt[[2]]*weight2
sum(pooled_pp)
pooled_pp <- pooled_pp / sum(pooled_pp)
pooled_pp[1:8]
sum(pooled_pp)
# make new table:
pooled <- data.table(
  Nn = merged$Nn,
  pp = pooled_pp
)
# sort:
pooled <- pooled[order(pp, decreasing = T)]
pooled

# save:
out_name <- paste(out, '.pp.dt.rds', sep='')
out_name
saveRDS(pooled, out_name)
