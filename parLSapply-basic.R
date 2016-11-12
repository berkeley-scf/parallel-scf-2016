library(parallel)
nCores <- 4  # to set manually 
cl <- makeCluster(nCores) 

source('rf.R')  # loads in data and looFit()

nSub <- 30
input <- seq_len(nSub) # same as 1:nSub but more robust

# clusterExport(cl, c('x', 'y')) # if the processes need objects
# from master's workspace (not needed here as no global vars used)


# need to load randomForest package within function
# when using par{L,S}apply
system.time(
	res <- parSapply(cl, input, looFit, Y, X, TRUE)
)
system.time(
	res2 <- sapply(input, looFit, Y, X)
)

res <- parLapply(cl, input, looFit, Y, X, TRUE)
