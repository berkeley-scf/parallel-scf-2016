library(parallel)

machines = c(rep("beren.berkeley.edu", 1),
    rep("gandalf.berkeley.edu", 1),
    rep("arwen.berkeley.edu", 2))
cl = makeCluster(machines)
cl

source('rf.R')  # loads in data and looFit()

n = 1e7
clusterExport(cl, c('n'))
  
res <- parSapply(cl, input, looFit, Y, X, TRUE)

result[1:5]

stopCluster(cl) # not strictly necessary
