source('rf.R')  # loads in data and looFit()

nSub <- 30
input <- seq_len(nSub) # same as 1:nSub but more robust

system.time(
	res <- mclapply(input, looFit, Y, X, mc.cores = nCores) 
)
