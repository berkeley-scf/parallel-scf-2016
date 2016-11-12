library(doParallel)  # uses parallel package, a core R package
# library(multicore); library(doMC) # alternative to parallel/doParallel
# library(Rmpi); library(doMPI) # to use Rmpi as the back-end

source('rf.R')  # loads in data and looFit()

looFit

nCores <- 4  # to set manually
registerDoParallel(nCores) 
# registerDoMC(nCores) # alternative to registerDoParallel
# cl <- startMPIcluster(nCores); registerDoMPI(cl) # when using Rmpi as the back-end

nSub <- 30  # do only first 30 for illustration

result <- foreach(i = 1:nSub) %dopar% {
	cat('Starting ', i, 'th job.\n', sep = '')
	output <- looFit(i, Y, X)
	cat('Finishing ', i, 'th job.\n', sep = '')
	output # this will become part of the out object
}
print(result[1:5])
