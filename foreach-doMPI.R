library(Rmpi)
library(doMPI)

cl = startMPIcluster()  # by default will start one fewer slave
# than elements in .hosts
                                        
source('rf.R')  # loads in data and looFit()

registerDoMPI(cl)
clusterSize(cl) # just to check

nSub <- 30  # do only first 30 for illustration

result <- foreach(i = 1:nSub, .packages = 'randomForest') %dopar% {
	cat('Starting ', i, 'th job.\n', sep = '')
	output <- looFit(i, Y, X)
	cat('Finishing ', i, 'th job.\n', sep = '')
	output # this will become part of the out object
}
print(result[1:5])

closeCluster(cl)

mpi.quit()
