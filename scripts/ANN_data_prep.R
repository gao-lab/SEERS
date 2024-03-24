# Prepare data sets for training and evaluating the neural network.

# the Nn expression table (RDS file) from previous step:
Nn_table = '/rd1/home/liangn/Works/SEERS-develop/experiment_Nn_log2expression/L5_log2expression.dt.rds'
# test data size:
test_n = 40000
# output directories and names:
train = '/rd1/home/liangn/Works/SEERS-develop/L5_log2expression_train_220528'
val = '/rd1/home/liangn/Works/SEERS-develop/L5_log2expression_val_220528'
test = '/rd1/home/liangn/Works/SEERS-develop/L5_log2expression_test_220528'

library(data.table)

# load data:
Nn.dt <- readRDS(Nn_table)
Nn.dt
str(Nn.dt)
# check Nn lengths:
table(nchar(Nn.dt$Nn))
# check Nn duplication:
which(duplicated(Nn.dt$Nn))

# sort by nDNA:
Nn.dt <- Nn.dt[order(nDNA, decreasing = T)]
Nn.dt

# search Nn:
which(Nn.dt$Nn == '')

# Initiate a Nn index pool:
i_pool <- 1:nrow(Nn.dt)

# Preserve the test set from top N45s with the most DNA:
test_i <- head(i_pool, test_n)
test_i
i_pool <- setdiff(i_pool, test_i)
test_data <- Nn.dt[test_i, c('Nn', 'nDNA', 'nuc.log2expression', 'cyt.log2expression')]
test_data
test_out_name <- paste(test, '.tsv', sep = '')
test_out_name
write.table(test_data, test_out_name, sep='\t', quote=F, row.names=F)

# Randomly draw 20% as validation set:
val_n <- length(Nn.dt$Nn) *0.2
val_n <- round(val_n)
val_n
val_i <- sample(i_pool, val_n, replace = F)
val_i <- sort(val_i, decreasing = F)
val_i
i_pool <- setdiff(i_pool, val_i)
val_data <- Nn.dt[val_i, c('Nn', 'nDNA', 'nuc.log2expression', 'cyt.log2expression')]
val_data
val_out_name <- paste(val, '.tsv', sep = '')
val_out_name
write.table(val_data, val_out_name, sep='\t', quote=F, row.names=F)

# training data:
train_data <- Nn.dt[i_pool, c('Nn', 'nDNA', 'nuc.log2expression', 'cyt.log2expression')]
train_data
train_out_name <- paste(train, '.tsv', sep = '')
train_out_name
write.table(train_data, train_out_name, sep='\t', quote=F, row.names=F)

# check duplication between sets:
str(train_data)
str(val_data)
str(test_data)
intersect(train_data$Nn, val_data$Nn)
intersect(train_data$Nn, test_data$Nn)
intersect(val_data$Nn, test_data$Nn)
