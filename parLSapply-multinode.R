library(parallel)

machines = c(rep("beren.berkeley.edu", 1),
    rep("gandalf.berkeley.edu", 1),
    rep("arwen.berkeley.edu", 2))
cl = makeCluster(machines)
cl

source('rf.R')  # loads in data and looFit()

nSub <- 30
input <- seq_len(nSub)

# not needed because Y and X are arguments,
# but would be needed if they were used as global variables
# clusterExport(cl, c('Y', 'X'))
  
result <- parSapply(cl, input, looFit, Y, X, TRUE)

result[1:5]

stopCluster(cl) # not strictly necessary
