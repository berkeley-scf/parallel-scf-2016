library(doSNOW)
machines = c(rep("beren.berkeley.edu", 1),
    rep("gandalf.berkeley.edu", 1),
    rep("arwen.berkeley.edu", 2))

cl = makeCluster(machines, type = "SOCK")
cl

registerDoSNOW(cl)

source('rf.R')  # loads in data and looFit()

nSub <- 30  # do only first 30 for illustration

result <- foreach(i = 1:nSub) %dopar% {
	cat('Starting ', i, 'th job.\n', sep = '')
	output <- looFit(i, Y, X)
	cat('Finishing ', i, 'th job.\n', sep = '')
	output # this will become part of the out object
}
print(result[1:5])


stopCluster(cl)  # good practice, but not strictly necessary
