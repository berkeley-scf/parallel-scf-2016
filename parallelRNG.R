# RNG with parallel apply functionality

library(parallel)
library(rlecuyer)
nSims <- 250
taskFun <- function(i){
	val <- runif(1)
	return(val)
}

nCores <- 4
RNGkind()
cl <- makeCluster(nCores)
iseed <- 0
clusterSetRNGStream(cl = cl, iseed = iseed)
RNGkind() # clusterSetRNGStream sets RNGkind as L'Ecuyer-CMRG
# but it doesn't show up here on the master
res <- parSapply(cl, 1:nSims, taskFun)
# now redo with same master seed to see results are the same
clusterSetRNGStream(cl = cl, iseed = iseed)
res2 <- parSapply(cl, 1:nSims, taskFun)
identical(res,res2)
stopCluster(cl)

# RNG with mclapply
library(parallel)
library(rlecuyer)
RNGkind("L'Ecuyer-CMRG")
res <- mclapply(seq_len(nSims), taskFun, mc.cores = nCores, 
    mc.set.seed = TRUE) 
# this also seems to reset the seed when it is run
res2 <- mclapply(seq_len(nSims), taskFun, mc.cores = nCores, 
    mc.set.seed = TRUE) 
identical(res,res2)

# reproducible RNG-based results in foreach
library(doRNG)
library(doParallel)
registerDoParallel(nCores)
registerDoRNG(seed = 0) 
result <- foreach(i = 1:20) %dopar% { 
	out <- mean(rnorm(1000)) 
}
registerDoRNG(seed = 0) 
result2 <- foreach(i = 1:20) %dopar% { 
	out <- mean(rnorm(1000)) 
}
identical(result,result2)
