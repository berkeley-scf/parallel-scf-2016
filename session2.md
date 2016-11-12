% Workshop on Parallelization and Campus Computing Resources (Part 2)
% November 15, 2016
% Chris Paciorek, Department of Statistics and Berkeley Research Computing, UC Berkeley

# This workshop

Session 1 on November 8 gave an overview of computing and data storage resources available through UC Berkeley and information on using the SCF and Savio Linux clusters. 

Session 2 on November 15 covers strategies for parallelizing your work, key concepts in implementing parallelization, and details of using parallelization in Python, R, MATLAB and C++. 

Savio is the campus Linux high-performance computing cluster, run by [Berkeley Research Computing](http://research-it.berkeley.edu/programs/berkeley-research-computing).

This tutorial assumes you have a working knowledge of bash and a scripting language such as Python, R, MATLAB, or Julia. 

Materials for this tutorial, including the Markdown file and associated code files that were used to create this document are available on Github at https://github.com/berkeley-scf/parallel-scf-2016.  You can download the files by doing a git clone from a terminal window on a UNIX-like machine, as follows:
```
git clone https://github.com/berkeley-scf/parallel-scf-2016
```

The materials are also available as a [zip file](https://github.com/berkeley-scf/parallel-scf-2016/archive/master.zip).


This material by Christopher Paciorek is licensed under a Creative Commons Attribution 3.0 Unported License.

# Learning resources and links

This workshop is based largely on already-prepared SCF and BRC (Berkeley Research Computing) material and other documentation that you can look at for more details:


Information on parallel coding (for Session 2)

Session 2 is essentially a subset/rearrangement of material in these SCF tutorials.

 - [Tutorial on shared memory parallel processing](https://github.com/berkeley-scf/tutorial-parallel-basics), in particular the [HTML overview](https://rawgit.com/berkeley-scf/tutorial-parallel-basics/master/parallel-basics.html)
 - [Tutorial on distributed memory parallel processing](https://github.com/berkeley-scf/tutorial-parallel-distributed), in particular the [HTML overview](https://rawgit.com/berkeley-scf/tutorial-parallel-distributed/master/parallel-dist.html)


# Taxonomy of parallel processing

There are two basic flavors of parallel processing (leaving aside
GPUs): distributed memory and shared memory. With shared memory, multiple
processors (which I'll call cores) share the same memory. With distributed
memory, you have multiple nodes, each with their own memory. You can
think of each node as a separate computer connected by a fast network. 

## Some useful terminology:

  - *cores*: We'll use this term to mean the different processing
units available on a single node.
  - *nodes*: We'll use this term to mean the different computers,
each with their own distinct memory, that make up a cluster or supercomputer.
  - *processes*: computational tasks executing on a machine; multiple
processes may be executing at once. A given program may start up multiple
processes at once. Ideally we have no more processes than cores on
a node.
  - *threads*: multiple paths of execution within a single process;
the OS sees the threads as a single process, but one can think of
them as 'lightweight' processes. Ideally when considering the processes
and their threads, we would have no more processes and threads combined
than cores on a node.
  - *forking*: child processes are spawned that are identical to
the parent, but with different process IDs and their own memory.
  - *sockets*: some of R's parallel functionality involves creating
new R processes (e.g., starting processes via *Rscript*) and
communicating with them via a communication technology called sockets.

## Shared memory

For shared memory parallelism, each core is accessing the same memory
so there is no need to pass information (in the form of messages)
between different machines. But in some programming contexts one needs
to be careful that activity on different cores doesn't mistakenly
overwrite places in memory that are used by other cores.

Two standard forms of parallelization on a single machine are: 

  - threaded code
     - openMP is the standard protocol for having pieces of a C/C++ program operate on multiple cores
     - modern, fast BLAS linear algebra packages such as openBLAS, MKL and ACML are threaded
  - multicore functionality 
     - this involves starting up new processes (computational engines) and dispatching computational tasks to them

### Threading

Threads are multiple paths of execution within a single process. If you are monitoring CPU
usage (such as with *top* from the Linux/Mac command line) and watching a job that is executing threaded code, you'll
see the process using more than 100% of CPU. When this occurs, the
process is using multiple cores, although it appears as a single process
rather than as multiple processes. 

## Distributed memory

Parallel programming for distributed memory parallelism requires passing
messages between the different nodes. The standard protocol for doing
this is MPI, of which there are various versions, including *openMPI*, which we'll use here.

You can write your own C/C++ code using MPI or use MPI via R or Python.
The R package *Rmpi* implements MPI in R. The *pbdR* packages for R also implement MPI as well as distributed linear algebra. Python has a package *mpi4py* that allows use of MPI within Python.

More simply, in both R and Python, there are easy ways to do embarrassingly parallel calculations (such as simple parallel for loops) across multiple machines, with MPI and similar tools used behind the scenes to manage the worker processes.

In summary, some types of distributed memory parallelization are:
 - simple parallelization of embarrassingly parallel computations (in R, Python, and Matlab) without writing code that explicitly uses MPI;
 - distributed linear algebra using the *pbdR* front-end to the *ScaLapack* package; and
 - using MPI explicitly (in R, Python and C).


## Other type of parallel processing

We won't cover either of these in this material.

### GPUs

GPUs (Graphics Processing Units) are processing units originally designed
for rendering graphics on a computer quickly. This is done by having
a large number of simple processing units for massively parallel calculation.
The idea of general purpose GPU (GPGPU) computing is to exploit this
capability for general computation. 

In spring 2016, I gave a [workshop on using GPUs](http://statistics.berkeley.edu/computing/gpu).

### Spark and Hadoop

Spark and Hadoop are systems for implementing computations in a distributed
memory environment, using the MapReduce approach. In fall 2014, I gave a [workshop on using Spark](http://statistics.berkeley.edu/computing/gpu).

# Parallelization strategies

The following are some basic principles/suggestions for how to parallelize
your computation.

Should I use one machine/node or many machines/nodes?

 - If you can do your computation on the cores of a single node using
shared memory, that will be faster than using the same number of cores
(or even somewhat more cores) across multiple nodes. Similarly, jobs
with a lot of data/high memory requirements that one might think of
as requiring Spark or Hadoop may in some cases be much faster if you can find
a single machine with a lot of memory.
 - That said, if you would run out of memory on a single node, then you'll
need to use distributed memory.

What level or dimension should I parallelize over?

 - If you have nested loops, you generally only want to parallelize at
one level of the code. That said, there may be cases in which it is
helpful to do both. Keep in mind whether your linear algebra is being
threaded. Often you will want to parallelize over a loop and not use
threaded linear algebra.
 - Often it makes sense to parallelize the outer loop when you have nested
loops.
 - To parallelize over multiple levels, it's usually best to 'flatten' the 
indexing and not explicitly do nested parallelization.
 - You generally want to parallelize in such a way that your code is
load-balanced and does not involve too much communication. 

How do I balance communication overhead with keeping my cores busy?

 - If you have very few tasks, particularly if the tasks take different
amounts of time, often some processors will be idle and your code
poorly load-balanced.
 - If you have very many tasks and each one takes little time, the communication
overhead of starting and stopping the tasks will reduce efficiency.

Should multiple tasks be pre-assigned to a process (i.e., a worker) (sometimes called *prescheduling*) or should tasks
be assigned dynamically as previous tasks finish? 

 - Basically if you have many tasks that each take similar time, you
want to preschedule the tasks to reduce communication. If you have few tasks
or tasks with highly variable completion times, you don't want to
preschedule, to improve load-balancing.
 - For R in particular, some of R's parallel functions allow you to say whether the 
tasks should be prescheduled. E.g., the *mc.preschedule* argument in *mclapply* and
the *.scheduling* argument in *parLapply*.


# Basic parallelized loops/maps/apply

## Overview

All of the functionality discussed here applies *only* if the iterations/loops of your calculations can be done completely separately and do not depend on one another. This scenario is called an *embarrassingly parallel* computation.  So coding up the evolution of a time series or a Markov chain is not possible using these tools. However, bootstrapping, random forests, simulation studies, cross-validation
and many other statistical methods can be handled in this way.

Most languages you'll encounter (in particular *functional* languages) will have the ability to operate a single function on multiple input values, thereby generating multiple tasks. This is often called a *map* operation, though in R one would call it an *apply*. 

Usually there is a master process that manages the mapping, including dispatching individual tasks to the workers and collecting all the results.

The key challenges that arise are:

 - starting up the worker processes and making sure the master process is communicating with them,
 - making sure that the workers have the necessary data and have loaded needed packages,
 - handling additional arguments to the function being used,
 - ensuring use of distinct random numbers in the different tasks, and
 - debugging code that operates in another process.

## Parallel loops and *apply* functions in R

Demo code below is also in various R files in the repository.

### Parallel for loops with *foreach*

A simple way to exploit parallelism in R  is to use the *foreach* package to do a for loop in parallel.

The *foreach* package provides a *foreach* command that
allows you to do this easily. *foreach* can use a variety of
parallel ``back-ends''. For our purposes, the main one is use of the *parallel* package to use shared
memory cores. When using *parallel* as the
back-end, you should see multiple processes (as many as you registered;
ideally each at 100%) when you  monitor CPU usage. The multiple processes
are created by forking or using sockets. 

```
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
```

(Note that the printed statements from `cat` are not showing up in the creation of this document but should show if you run the code.)

 Note that *foreach*
also provides functionality for collecting and managing
the results to avoid some of the bookkeeping
you would need to do if writing your own standard for loop.
The result of *foreach* will generally be a list, unless 
we request the results be combined in different way, as we do here using `.combine = c`.

You can debug by running serially using *%do%* rather than
*%dopar%*. Note that you may need to load packages within the
*foreach* construct to ensure a package is available to all of
the calculations.


### Parallel apply functionality

The *parallel* package has the ability to parallelize the various
*apply* functions (apply, lapply, sapply, etc.). It's a bit hard to find the [vignette for the parallel package](http://stat.ethz.ch/R-manual/R-devel/library/parallel/doc/parallel.pdf)
because parallel is not listed as one of
the contributed packages on CRAN (it gets installed with R by default).

We'll consider parallel lapply and sapply. These rely on having started a cluster using *cluster*, which  uses the PSOCK mechanism as in the SNOW package - starting new jobs via *Rscript* 
and communicating via a technology called sockets.

```
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
```

Here the miniscule user time is probably because the time spent in the worker processes is not counted at the level of the overall master process that dispatches the workers.

For help with these functions and additional related parallelization functions (including *parApply*), see `help(clusterApply)`.

*mclapply* is an alternative that uses forking to start up the worker processes.

```
source('rf.R')  # loads in data and looFit()

nSub <- 30
input <- seq_len(nSub) # same as 1:nSub but more robust

system.time(
	res <- mclapply(input, looFit, Y, X, mc.cores = nCores) 
)
```

Note that some R packages can directly interact with the parallelization
packages to work with multiple cores. E.g., the *boot* package
can make use of the *parallel* package directly. 

### Loading packages and accessing global variables within your parallel tasks

Whether you need to explicitly load packages and export global variables from the master process to the parallelized worker processes depends on the details of how you are doing the parallelization.

In R, with *foreach* with the *doParallel* backend, parallel *apply* statements (starting the cluster via *makeForkCluster*, instead of the usual *makeCluster*), and *mclapply*, packages and global variables in the main R process are automatically available to the worker tasks without any work on your part. This is because all of these approaches fork the original R process, thereby creating worker processes with the same state as the original R process. Interestingly, this means that global variables in the forked worker processes are just references to the objects in memory in the original R process. So the additional processes do not use additional memory for those objects (despite what is shown in *top*) and there is no time involved in making copies. However, if you modify objects in the worker processes then copies are made. 

In contrast, with parallel *apply* statements when starting the cluster using the standard *makeCluster* (which sets up a so-called *PSOCK* cluster, starting the R worker processes via *Rscript*), one needs to load packages within the code that is executed in parallel. In addition one needs to use *clusterExport* to tell R which objects in the global environment should be available to the worker processes. This involves making as many copies of the objects as there are worker processes, so one can easily exceed the physical memory (RAM) on the machine if one has large objects, and the copying of large objects will take time. 

## Parallel looping in Python

Demo code below is also in *ipython-parallel.py*.

I'll cover iPython parallel functionality, which allows one to parallelize on a single machine (discussed here) or across multiple machines (see the tutorial on distributed memory parallelization). There are a variety of other approaches one could use, of which I discuss two (the *pp* and *multiprocessing* packages) in the Appendix.

First we need to start our worker engines.

```
ipcluster start -n 4 &
sleep 45
```

Here we'll generate some fake data to fit a random forest model to and then use leave-one-out cross-validation to assess the model, parallelizing over the individual held-out observations.

```
import numpy as np
np.random.seed(0)
n = 500
p = 50
X = np.random.normal(0, 1, size = (n, p))
Y = X[: , 0] + pow(abs(X[:,1] * X[:,2]), 0.5) + X[:,1] - X[:,2] + np.random.normal(0, 1, n)

def looFit(index, Ylocal, Xlocal):
    rf = rfr(n_estimators=100)
    fitted = rf.fit(np.delete(Xlocal, index, axis = 0), np.delete(Ylocal, index))
    pred = rf.predict(np.array([Xlocal[index, :]]))
    return(pred[0])

from ipyparallel import Client
c = Client()
c.ids

dview = c[:]
dview.block = True
dview.apply(lambda : "Hello, World")

lview = c.load_balanced_view()
lview.block = True

dview.execute('from sklearn.ensemble import RandomForestRegressor as rfr')
dview.execute('import numpy as np')
mydict = dict(X = X, Y = Y, looFit = looFit)
dview.push(mydict)

nSub = 50  # for illustration only do a subset

# need a wrapper function because map() only operates on one argument
def wrapper(i):
    return(looFit(i, Y, X))

import time
time.time()
pred = lview.map(wrapper, range(nSub))
time.time()

print(pred[0:10])

# import pylab
# import matplotlib.pyplot as plt
# plt.plot(Y, pred, '.')
# pylab.show()
```

Finally we stop the worker engines:

```
ipcluster stop
```

### Loading packages and accessing global variables within your parallel tasks

In iPython parallel, you need to do some work to ensure that data and packages are available on the workers. 

 - Package loading needs to be done by an explicit parallel call as seen above. 
 - Since map() only operates on a single input, we write a wrapper function that operates only on the index value and that calls the real function that runs on the workers and uses data objects previously broadcast to the workers. 

# Basic parallelized loops/maps/apply across multiple machines

## Distributed foreach and apply in R

### foreach with the *doMPI* and *doSNOW* backends

Just as we used *foreach* in a shared memory context, we can
use it in a distributed memory context as well, and R will handle
everything behind the scenes for you. 

#### *doMPI*

Start R through the *mpirun* command as discussed above, either
as a batch job or for interactive use. We'll only ask for 1 process
because the worker processes will be started automatically from within R (but using the machine names information passed to mpirun).

When using this within SLURM, we don't need to specify the machines to be used because SLURM and MPI are integrated.

```
mpirun R CMD BATCH -q --no-save doMPI.R doMPI.out
```

If one were doing this outside of SLURM, one may need to do this so that MPI knows the machines (and number of cores per machine) that are available:
```
mpirun -machinefile .hosts R CMD BATCH -q --no-save doMPI.R doMPI.out
mpirun -machinefile .hosts R --no-save
```

where *.hosts* would look something like this:
```
arwen.berkeley.edu slots=3
beren.berkeley.edu slots=2
```

Here's R code for using *Rmpi* as the back-end to *foreach*.
If you call *startMPIcluster* with no arguments, it will start
up one fewer worker processes than the number of hosts times slots given to mpirun
so your R code will be more portable. 

```
library(Rmpi)
library(doMPI)

cl = startMPIcluster()  # by default will start one fewer slave
# than elements in .hosts
                                        
source('rf.R')  # loads in data and looFit()

registerDoMPI(cl)
clusterSize(cl) # just to check

nSub <- 30  # do only first 30 for illustration

result <- foreach(i = 1:nSub) %dopar% {
	cat('Starting ', i, 'th job.\n', sep = '')
	output <- looFit(i, Y, X)
	cat('Finishing ', i, 'th job.\n', sep = '')
	output # this will become part of the out object
}
print(result[1:5])

closeCluster(cl)

mpi.quit()
```


A caution concerning Rmpi/doMPI: when you invoke `startMPIcluster()`,
all the slave R processes become 100% active and stay active until
the cluster is closed. In addition, when *foreach* is actually
running, the master process also becomes 100% active. So using this
functionality involves some inefficiency in CPU usage. This inefficiency
is not seen with a sockets cluster (Section 3.1.4) nor when using other
Rmpi functionality - i.e., starting slaves with *mpi.spawn.Rslaves*
and then issuing commands to the slaves.

If you specified `-np` with more than one process then as with the C-based
MPI job above, you can control the threading via OMP_NUM_THREADS
and the -x flag to *mpirun*. Note that this only works when the
R processes are directly started by *mpirun*, which they are
not if you set -np 1. The *maxcores* argument to *startMPIcluster()*
does not seem to function (perhaps it does on other systems).

Sidenote: You can use *doMPI* on a single node, which might be useful for avoiding
some of the conflicts between R's forking functionality and openBLAS that
can cause R to hang when using *foreach* with *doParallel*.

#### *doSNOW*

The *doSNOW* backend has the advantage that it doesn't need to have MPI installed on the system. MPI can be tricky to install and keep working, so this is an easy approach to using *foreach* across multiple machines.

Simply start R as you usually would. 

Here's R code for using *doSNOW* as the back-end to *foreach*. Make sure to use the `type = "SOCK"` argument or *doSNOW* will actually use MPI behind the scenes. 

```
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
```

#### Loading packages and accessing variables within your parallel tasks

When using *foreach* with multiple machines, you need to use the *.packages* argument (or load the package in the code being run in parallel) to load any packages needed in the code. You do not need to explicitly export variables from the master process to the workers. Rather, *foreach* determines which variables in the global environment of the master process are used in the code being run in parallel and makes copies of those in each worker process. Note that these variables are read-only on the workers and cannot be modified (if you try to do so, you'll notice that *foreach* actually did not make copies of the variables that your code tries to modify). 

### 3.1.4) Using parallel apply with makeCluster on multiple nodes

One can also set up a cluster with the worker processes communicating via sockets. You just need to specify
a character vector with the machine names as the input to *makeCluster()*. A nice thing about this is that it doesn't involve any of the complications of working with needing MPI installed.

```
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
```

Note the use of *clusterExport*, needed to make variables in the master process available to the workers; this involves making a copy of each variable for each worker process. You'd also need to load any packages used in the code being run in parallel in that code. 


## Distributed parallel looping in Python

One can use iPython's parallelization tools in a context with multiple nodes, though the setup to get the worker processes is a bit more involved when you have multiple nodes. For details on using iPython parallel on a single node, see the [parallel basics tutorial appendix](https://github.com/berkeley-scf/tutorial-parallel-basics). 

If we are using the SLURM scheduling software, here's how we start up the worker processes:

```
ipcontroller --ip='*' &
sleep 25
# next line will start as many ipengines as we have SLURM tasks 
#   because srun is a SLURM command
srun ipengine &  
sleep 45  # wait until all engines have successfully started
```

We can then run iPython to split up our computational tasks across the engines, using the same code as we used in the single-node context above.

To finish up, we need to shut down the cluster of workers:
```
ipcluster stop
```

To start the engines in a context outside of using slurm (provided all machines share a filesystem), you should be able ssh to each machine and run `ipengine &` for as many worker processes as you want to start as follows. In some, but not all cases (depending on how the network is set up) you may not need the `--location` flag. 

```
ipcontroller --ip='*' --location=URL_OF_THIS_MACHINE &
sleep 25
nengines=8
ssh other_host "for (( i = 0; i < ${nengines}; i++ )); do ipengine & done"
sleep 45  # wait until all engines have successfully started
```

# Random number generation (RNG) in parallel 

The key thing when thinking about random numbers in a parallel context
is that you want to avoid having the same 'random' numbers occur on
multiple processes. On a computer, random numbers are not actually
random but are generated as a sequence of pseudo-random numbers designed
to mimic true random numbers. The sequence is finite (but very long)
and eventually repeats itself. When one sets a seed, one is choosing
a position in that sequence to start from. Subsequent random numbers
are based on that subsequence. All random numbers can be generated
from one or more random uniform numbers, so we can just think about
a sequence of values between 0 and 1. 

The worst thing that could happen is that one sets things up in such
a way that every process is using the same sequence of random numbers.
This could happen if you mistakenly set the same seed in each process,
e.g., using *set.seed(mySeed)* in R on every process.

The naive approach is to use a different seed for each process. E.g.,
if your processes are numbered `id = 1,2,...,p`  with a variable *id* that is  unique
to a process, setting the seed to be the value of *id* on each process. This is likely
not to cause problems, but raises the danger that two (or more sequences)
might overlap. For an algorithm with dependence on the full sequence,
such as an MCMC, this probably won't cause big problems (though you
likely wouldn't know if it did), but for something like simple simulation
studies, some of your 'independent' samples could be exact replicates
of a sample on another process. Given the period length of the default
generators in R, Matlab and Python, this is actually quite unlikely,
but it is a bit sloppy.

One approach to avoid the problem is to do all your RNG on one process
and distribute the random deviates, but this can be infeasible with
many random numbers.

More generally to avoid this problem, the key is to use an algorithm
that ensures sequences that do not overlap.


## Ensuring separate sequences in R

In R, the  *rlecuyer* package deals with this (*rsprng* used to but it is no longer on CRAN).
The L'Ecuyer algorithm has a period of $2^{191}$, which it divides
into subsequences of length $2^{127}$. 

The code below is also in *parallelRNG.R*.

### With the parallel package

Here's how you initialize independent sequences on different processes
when using the *parallel* package's parallel apply functionality
(illustrated here with *parSapply*).

```
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
```



When using *mclapply*, you can use the *mc.set.seed* argument
as follows (note that *mc.set.seed* is TRUE by default, so you
should get different seeds for the different processes by default),
but one needs to invoke `RNGkind("L'Ecuyer-CMRG")`
to get independent streams via the L'Ecuyer algorithm.

```
library(parallel)
library(rlecuyer)
RNGkind("L'Ecuyer-CMRG")
res <- mclapply(seq_len(nSims), taskFun, mc.cores = nCores, 
    mc.set.seed = TRUE) 
# this also seems to reset the seed when it is run
res2 <- mclapply(seq_len(nSims), taskFun, mc.cores = nCores, 
    mc.set.seed = TRUE) 
identical(res,res2)
```

The documentation for *mcparallel* gives more information about
reproducibility based on *mc.set.seed*.


###  With foreach


#### Getting independent streams

One question is whether *foreach* deals with RNG correctly. This
is not documented, but the developers (Revolution Analytics) are well
aware of RNG issues. Digging into the underlying code reveals that
the *doParallel* backend invokes *mclapply*
and sets *mc.set.seed* to TRUE by default. This suggests that
the discussion above r.e. *mclapply* holds for *foreach*
as well, so you should do `RNGkind("L'Ecuyer-CMRG")`
before your foreach call. 

#### Ensuring reproducibility

While using *foreach* as just described should ensure that the
streams on each worker are are distinct, it does not ensure reproducibility
because task chunks may be assigned to workers differently in different
runs and the substreams are specific to workers, not to tasks. 

For backends other than *doMPI*, such as *doParallel*, there is a package
called *doRNG* that ensures that *foreach* loops are reproducible. (For *doMPI* you simply pass `.options.mpi = list(seed = your_seed_value_here)` as an additional argument to *foreach*.)

Here's how you do it:

```
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
```

You can ignore the warnings about closing unused connections printed out above.

## RNG in Python

Python uses the Mersenne-Twister generator. If you're using the RNG
in *numpy/scipy*, you can set the seed using `numpy.random.seed` or `scipy.random.seed`.
The advice I'm seeing online in various Python forums is to just set
separate seeds, so it appears the Python is a bit behind R and Matlab here.
 There is a function *random.jumpahead* that
allows you to move the seed ahead as if a given number of random numbers
had been generated, but this function will not be in Python 3.x, so
I won't suggest using it. 




# MPI for distributed memory computation 

##  MPI Overview

There are multiple MPI implementations, of which *openMPI* and
*mpich* are very common. *openMPI* is on the SCF and Savio, and we'll use that.

In MPI programming, the same code runs on all the machines. This is
called SPMD (single program, multiple data). As we saw a bit with the pbdR code, one
invokes the same code (same program) multiple times, but the behavior
of the code can be different based on querying the rank (ID) of the
process. Since MPI operates in a distributed fashion, any transfer
of information between processes must be done explicitly via send
and receive calls (e.g., *MPI_Send*, *MPI_Recv*, *MPI_Isend*,
and *MPI_Irecv*). (The ``MPI_'' is for C code; C++ just has
*Send*, *Recv*, etc.)

The latter two of these functions (*MPI_Isend* and *MPI_Irecv*)
are so-called non-blocking calls. One important concept to understand
is the difference between blocking and non-blocking calls. Blocking
calls wait until the call finishes, while non-blocking calls return
and allow the code to continue. Non-blocking calls can be more efficient,
but can lead to problems with synchronization between processes. 

In addition to send and receive calls to transfer to and from specific
processes, there are calls that send out data to all processes (*MPI_Scatter*),
gather data back (*MPI_Gather*) and perform reduction operations
(*MPI_Reduce*).

Debugging MPI code can be tricky because communication
can hang, error messages from the workers may not be seen or readily
accessible, and it can be difficult to assess the state of the worker
processes. 

## Basic syntax for MPI in C


Here's a basic hello world example  The code is also in *mpiHello.c*.

```
// see mpiHello.c
#include <stdio.h> 
#include <math.h> 
#include <mpi.h>

int main(int argc, char* argv) {     
	int myrank, nprocs, namelen;     
	char process_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);     
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);   
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);          
	MPI_Get_processor_name(process_name, &namelen);            
	printf("Hello from process %d of %d on %s\n", 
		myrank, nprocs, process_name);
    MPI_Finalize();     
	return 0; 
} 
```

There are C (*mpicc*) and C++ (*mpic++*) compilers for MPI programs (*mpicxx* and *mpiCC* are synonyms).
I'll use the MPI C++ compiler
even though the code is all plain C code.


```
mpicxx mpiHello.c -o mpiHello
cat .hosts # what hosts do I expect it to run on?
mpirun -machinefile .hosts -np 4 mpiHello
```



To actually write real MPI code, you'll need to go learn some of the
MPI syntax. See *quad_mpi.c* and *quad_mpi.cpp*, which
are example C and C++ programs (for approximating an integral via
quadrature) that show some of the basic MPI functions. Compilation
and running are as above:

```
mpicxx quad_mpi.cpp -o quad_mpi
mpirun -machinefile .hosts -np 4 quad_mpi
```


## Using MPI from Python and R

Both R (via Rmpi and pbdR) and Python (via mpi4py) allow you to make MPI calls within R and Python. 

Here's some basic use of MPI within Python.

```{r, mpi4py, engine='python', eval=FALSE}
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

# simple print out Rank & Size
id = comm.Get_rank()
print "Of ", comm.Get_size() , " workers, I am number " , id, "."

def f(id, n):
    np.random.seed(id)
    return np.mean(np.random.normal(0, 1, n))

n = 1000000
result = f(id, n)


output = comm.gather(result, root = 0)

if id == 0:
    print output
```

To run the code, we start Python through the mpirun command as done previously.

```
mpirun -machinefile .hosts -np 4 python example-mpi.py 
```

More generally, you can send, receive, broadcast, gather, etc. as with MPI itself.

*mpi4py* generally does not work interactively.

The feel of using MPI via Rmpi is somewhat similar.
