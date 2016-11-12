% Workshop on Parallelization and Campus Computing Resources (Part 1)
% November 8, 2016
% Chris Paciorek, Department of Statistics and Berkeley Research Computing, UC Berkeley

# This workshop

This workshop gives an overview of computing and data storage resources available through UC Berkeley and information on using the SCF and Savio Linux clusters. 

Session 2 on November 15 will cover strategies for parallelizing your work and using parallelization in Python, R, MATLAB and C++. 

Savio is the campus Linux high-performance computing cluster, run by [Berkeley Research Computing](http://research-it.berkeley.edu/programs/berkeley-research-computing).

This tutorial assumes you have a working knowledge of bash and a scripting language such as Python, R, MATLAB, or Julia. 

Materials for this tutorial, including the Markdown file and associated code files that were used to create this document are available on Github at https://github.com/berkeley-scf/parallel-scf-2016.  You can download the files by doing a git clone from a terminal window on a UNIX-like machine, as follows:
```
git clone https://github.com/berkeley-scf/parallel-scf-2016
```

The materials are also available as a [zip file](https://github.com/berkeley-scf/parallel-scf-2016/archive/master.zip).


This material by Christopher Paciorek is licensed under a Creative Commons Attribution 3.0 Unported License.

# Learning resources and links

This workshop is based in part on already-prepared SCF and BRC (Berkeley Research Computing) material and other documentation that you can look at for more details:

Information on campus resources

 - [Instructions for using the SCF SLURM (high-priority) cluster](http://statistics.berkeley.edu/computing/servers/cluster-high)
 - [Instructions for using the SCF SGE (standard) cluster](http://statistics.berkeley.edu/computing/servers/cluster)
 - [Instructions for using the Savio campus Linux cluster](http://research-it.berkeley.edu/services/high-performance-computing)
 - [Tutorial on basic usage of Savio](https://github.com/ucberkeley/savio-training-intro-2016), in particular the [HTML overview](https://rawgit.com/ucberkeley/savio-training-intro-2016/master/intro.html)
 - [Tutorial on parallelization on Savio](https://github.com/ucberkeley/savio-training-parallel-2016), in particular the [HTML overview](https://rawgit.com/ucberkeley/savio-training-parallel-2016/master/parallel.html)

Information on parallel coding (for Session 2)

 - [Tutorial on shared memory parallel processing](https://github.com/berkeley-scf/tutorial-parallel-basics), in particular the [HTML overview](https://rawgit.com/berkeley-scf/tutorial-parallel-basics/master/parallel-basics.html)
 - [Tutorial on distributed memory parallel processing](https://github.com/berkeley-scf/tutorial-parallel-distributed), in particular the [HTML overview](https://rawgit.com/berkeley-scf/tutorial-parallel-distributed/master/parallel-dist.html)

# Outline

 - Statistics and campus computing resources
      - computation
      - data storage
 - Parallelization terminology and concepts
 - Basic job submission in SLURM on the SCF
     - SCF clusters overview
     - Accessing software via modules
     - Basic job submission
     - Monitoring jobs and cluster status
     - Interactive jobs
     - Parallel jobs
     - Using environment variables
     - GPU jobs
 - Job submission on Savio
     - Logging in
     - Accessing software via modules
     - Accounts and partitions
     - Basic job submission
     - Monitoring jobs and cluster status
     - Interactive jobs
     - Using environment variables
     - Parallel jobs
     - Low-priority queue
     - GPU jobs
     - Spark jobs
 - Data transfer
      - Using standard tools (SCF/SFTP/GUIs)
      - Using Globus
      - Berkeley Box and bDrive 
 - Getting help

# Computational resources

 - [Statistics cluster](http://statistics.berkeley.edu/computing/servers)
    - 'High-priority' cluster: 4 nodes x 24 cores/node; 64 GB RAM per node; SLURM queueing; 14 day time limit
    - Standard cluster: 8 nodes x 32 cores/node; 256 GB RAM per node; SGE queueing (to be switched to SLURM in coming months); 28 day time limit
    - standard software: Python, R, MATLAB, Julia
    - 4 GPUs but some obtained for particular research groups
 - [Savio campus cluster](http://research-it.berkeley.edu/services/high-performance-computing)
    - ~6600 nodes across ~330 nodes, 64 GB RAM on most nodes but up to 512 GB RAM on high-memory nodes
    - access through:
         - faculty computing allowance 
              - CPU-only and GPU nodes
              - 3 day time limit
         - Statistics department condo
              - 2 CPU-only nodes, 24 cores per node
              - 1 GPU node, 4 GPUs (currently with priority access for one research group)
              - no time limit at the moment
    - Python, R, MATLAB, Spark
 - [NSF XSEDE network](https://www.xsede.org) of supercomputers
    - Bridges supercomputer for big data computing, including Spark
    - many other clusters/supercomputers
    - access:
          - can get initial access via brc@berkeley.edu       
          - follow up with a start-up allocation (one-page, always approved)
          - finally, research grants possible (several page process)

**Big picture: if you don't have enough computing resources, don't give up and work on a smaller problem, talk to us at consult@stat.berkeley.edu.**

# Data storage resources

- SCF
     - tens-hundreds of GB potentially available on SCF home directories
     - hundreds of GB to TB potentially available on scratch on SCF
     - total disk ~ 9 TB
- Savio
     - 'Unlimited' storage in scratch on Savio (old files purged eventually)
     - Total scratch space ~ 850 TB
     - Savio "condo" storage: $7000 for 25 TB for 5 years
- UC Berkeley cloud providers
     - Unlimited storage on Box through Berkeley (15 GB file limit)
     - Unlimited storage on Google Drive (bDrive) through Berkeley (5 TB file limit)

# Parallelization terminology and concepts

  - *cores*: We'll use this term to mean the different processing
units available on a single node.
  - *nodes*: We'll use this term to mean the different computers,
each with their own distinct memory, that make up a cluster or supercomputer.
  - *processes* or *SLURM tasks*: computational instances executing on a machine; multiple
processes may be executing at once. Ideally we have no more processes than cores on
a node.
  - *threads*: multiple paths of execution within a single process;
the OS sees the threads as a single process, but one can think of
them as 'lightweight' processes. Ideally when considering the processes
and their threads, we would have no more processes and threads combined
than cores on a node.
 - *computational tasks*: We'll use this to mean the independent computational units that make up the job you submit
    - each *process* or *SLURM task* might carry out one computational task or might be assigned multiple tasks sequentially or as a group.

# Job competition and scheduling on the SCF and Savio clusters

 - goals:
     - allow users to share the CPUs equitably
     - efficiently use the CPUs to maximum potential
     - allow large jobs to run
     - allow users to submit many jobs and have the scheduler manage them
 - jobs with various requirements
     - time length
     - number of cores
 - current scheduling policies
     - fairshare for queue
     - once running, only subject to time limits
     - backfilling and (on old SCF cluster) resource reservations
 - time limits 
     - very helpful for the scheduler
     - required on Savio
     - not required (for now) on SCF, but please set time limits if you are submitting dozens/hundreds/more jobs


# SCF clusters

SLURM (new) cluster

 - submission nodes: all SCF Linux machines
 - compute nodes: scf-sm20, scf-sm21, scf-sm22, scf-sm23
 - 96 total cores
 - 14 day time limit by default
 - if submitting many jobs (say more than 30) please try to give a rough time limit but be conservative

SGE (old) cluster

 - submission nodes: all SCF Linux machines
 - compute nodes: scf-sm0[0123], scf-sm1[0123]
 - 256 total cores
 - 28 day maximum time limit (needs to be requested with job)
 - to transition to SLURM in coming months

# SCF: Accessing software via modules

Savio and other systems make use of Linux environment modules extensively to provide software such that you can access different versions of software and access software with conflicting requirements.

We're starting to make use of modules on the SCF.

```
module list
module avail

module unload python
module load python/2
module list

module switch python/2 python/3
```

At this point most modules on the SCF are for machine learning software that can use GPUs.
  
# SLURM: basic batch job submission

Let's see how to submit a simple job. 

Here's an example job script (*job-basic-scf.sh*) that I'll run. 

```
#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Wall clock limit (3 minutes here):
#SBATCH --time=00:03:00
#
## Command(s) to run:
python calc.py >& calc.out
```

By default this will request only one core for the job.

To submit the job:

```
sbatch job-basic-scf.sh
```

Note that the JOB_ID is printed as the output of this call.

We could also include the SLURM flags in the submission script. Here's the simplified script:

```
#!/bin/bash
python calc.py >& calc.out
```

and here is the job submission:

```
sbatch --job-name=test --time=00:03:00 job-basic-scf.sh
```

# SCF: Monitoring jobs and cluster status

`squeue` allows us to monitor our jobs and the general state of the cluster and of queued jobs.

```
squeue -j JOB_ID

squeue -u ${USER}

# to see a bunch of useful information on all jobs
alias sq='squeue -o "%.7i %.9P %.20j %.8u %.2t %l %.9M %.5C %.8r %.6D %R %p %q"'
sq
```

# SCF: Interactive jobs

To start an interactive session,

```
srun --pty /bin/bash
```

To use graphical interfaces, you need to do add an extra flag:

```
srun --pty --x11=first /bin/bash
matlab

# alternatively:
srun --pty --x11=first matlab
```

To connect to a specific node (e.g., to monitor another job or copy files to/from /tmp or /var/tmp on the node):

```
srun --pty -w scf-sm22 /bin/bash
```

# SCF: Parallel jobs

If you are submitting a job that uses multiple nodes, you'll need to carefully specify the resources you need. The key flags for use in your job script are:

 - `--cpus-per-task` (or `-c`): number of cpus to be used for each task
 - `--ntasks-per-node`: number of SLURM tasks (i.e., processes) one wants to run on each node
 - `--nodes` (or `-N`): number of nodes to use

In addition, in some cases it can make sense to use the `--ntasks` (or `-n`) option to indicate the total number of SLURM tasks and let the scheduler determine how many nodes and tasks per node are needed. 

Here's an example job script (see also *job-parallel-scf.sh*) for a parallel iPython job.

```
#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Number of nodes
#SBATCH --nodes=1
#
# Number of processors per node:
#SBATCH --ntasks-per-node=8
#
# Wall clock limit:
#SBATCH --time=00:05:00
#
## Command(s) to run:
ipcluster start -n $SLURM_NTASKS_PER_NODE &
sleep 40
export DATADIR=/scratch/users/paciorek/243/AirlineData
ipython parallel-analysis.py > parallel-analysis.pyout
```

# SCF: Using environment variables

When you write your code, you may need to specify information about the number of cores to use. SLURM will provide a variety of variables that you can use in your code so that it adapts to the resources you have requested rather than being hard-coded. 

Here are some of the variables that may be useful: SLURM_NTASKS, SLURM_NTASKS_PER_NODE, SLURM_CPUS_PER_TASK, SLURM_NODELIST, SLURM_NNODES.

To control your code and use the resources you've requested, you can use those variables in your job submission script (as above) or read those variables into your program.

For example:

```
import os                               ## Python
int(os.environ['SLURM_NTASKS'])         ## Python

as.numeric(Sys.getenv('SLURM_NTASKS'))  ## R

str2num(getenv('SLURM_NTASKS')))        ## MATLAB
```


# SCF: GPU jobs

The SCF has four GPUs. One of those is on scf-sm20 in the SLURM cluster. The other three were obtained by faculty members are reserved for priority access for their groups. In some cases we may be able to help you get access to those other GPUs.

To use the scf-sm20 GPU, you need the following flags in your srun/sbatch invocation (or your job submission script):

```
--partition=gpu -w scf-sm20-gpu --gres=gpu:1
```

Note that you can't use more than two CPUs with such a GPU jobs.

If you're using Tensorflow, Caffe, Theano, or Torch with GPU computation, you'll generally need to load the appropriate module to make the GPU-enabled software available:

```
module load tensorflow
module load caffe  # python 3 support
module load caffe/2016-10-05-py2 # python 2 support
module load torch
module load theano
```

If you load one of these, you'll see that CUDA and (in some cases) cuDNN are also loaded for you. 
 

# Savio: Logging in

To login, you need to have software on your own machine that gives you access to a UNIX terminal (command-line) session. These come built-in with Mac (see `Applications -> Utilities -> Terminal`). For Windows, some options include [PuTTY](http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html).

You also need to set up your smartphone or tablet with *Google Authenticator* to generate one-time passwords for you.

Here are instructions for [doing this setup, and for logging in](http://research-it.berkeley.edu/services/high-performance-computing/logging-savio).

Then to login:
```
ssh SAVIO_USERNAME@hpc.brc.berkeley.edu
```

Then enter XXXXXYYYYYY where XXXXXX is your PIN and YYYYYY is the one-time password. YYYYYY will be shown when you open your *Google authenticator* app on your phone/tablet.

One can then navigate around and get information using standard UNIX commands such as `ls`, `cd`, `du`, `df`, etc.

If you want to be able to open programs with graphical user interfaces:
```
ssh -Y SAVIO_USERNAME@hpc.brc.berkeley.edu
```

To display the graphical windows on your local machine, you'll need X server software on your own machine to manage the graphical windows. For Windows, your options include *eXceed* or *Xming* and for Mac, there is *XQuartz*.


# Savio: Accessing software via modules

A lot of software is available on Savio but needs to be loaded from the relevant software module before you can use it.

```
module list  # what's loaded?
module avail  # what's available
```

One thing that tricks people is that the modules are arranged in a hierarchical (nested) fashion, so you only see some of the modules as being available *after* you load the parent module. Here's how we see the Python packages that are available.

```
which python
python

module avail
module load python/2.7.8
which python
module avail
module load numpy
python 
# import numpy as np
```

Similarly, we can see that linear algebra, FFT, and HDF5/NetCDF software is available only after loading either the intel or gcc modules.

```
module load intel
module avail
module swap intel gcc
module avail
```


# Savio: Submitting jobs -- accounts and partitions

All computations are done by submitting jobs to the scheduling software that manages jobs on the cluster, called SLURM.

When submitting a job, the main things you need to indicate are the project account you are using (in some cases you might have access to multiple accounts such as an FCA and a condo) and the partition.

You can see what accounts you have access to and which partitions within those accounts as follows:

```
sacctmgr -p show associations user=${USER}
```

Here's an example of the output for a user who has access to an FCA, a condo, and a special partner account:
```
Cluster|Account|User|Partition|Share|GrpJobs|GrpTRES|GrpSubmit|GrpWall|GrpTRESMins|MaxJobs|MaxTRES|MaxTRESPerNode|MaxSubmit|MaxWall|MaxTRESMins|QOS|Def QOS|GrpTRESRunMins|
brc|co_stat|paciorek|savio2_bigmem|1||||||||||||savio_lowprio|savio_lowprio||
brc|co_stat|paciorek|savio2_gpu|1||||||||||||savio_lowprio,stat_gpu2_normal|stat_gpu2_normal||
brc|co_stat|paciorek|savio2_htc|1||||||||||||savio_lowprio|savio_lowprio||
brc|co_stat|paciorek|savio|1||||||||||||savio_lowprio|savio_lowprio||
brc|co_stat|paciorek|savio_bigmem|1||||||||||||savio_lowprio|savio_lowprio||
brc|co_stat|paciorek|savio2|1||||||||||||savio_lowprio,stat_normal|stat_normal||
brc|fc_paciorek|paciorek|savio2_gpu|1||||||||||||savio_debug,savio_normal|savio_normal||
brc|fc_paciorek|paciorek|savio2_htc|1||||||||||||savio_debug,savio_normal|savio_normal||
brc|fc_paciorek|paciorek|savio2_bigmem|1||||||||||||savio_debug,savio_normal|savio_normal||
brc|fc_paciorek|paciorek|savio2|1||||||||||||savio_debug,savio_normal|savio_normal||
brc|fc_paciorek|paciorek|savio|1||||||||||||savio_debug,savio_normal|savio_normal||
brc|fc_paciorek|paciorek|savio_bigmem|1||||||||||||savio_debug,savio_normal|savio_normal||
```

Because you are part of a condo, you'll notice that you have *low-priority* access to certain partitions. In addition to the Statistics nodes in our condo to which we have normal access, we can also burst beyond the condo and use other partitions at low-priority (see below).

In contrast, through my FCA, I have access to all the various partitions at normal priority, but of course I have to 'pay' for access through my FCA allotment.

# Savio: Basic job submission

Let's see how to submit a simple job. If your job will only use the resources on a single node, you can do the following. 


Here's an example job script that I'll run on the SCF condo.

```
#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=co_stat
#
# Partition:
#SBATCH --partition=savio2
#
# Wall clock limit (30 seconds here):
#SBATCH --time=00:00:30
#
## Command(s) to run:
module load python/3.2.3 numpy
python3 calc.py >& calc.out
```

Now let's submit and monitor the job:

```
sbatch job-basic-savio.sh
```

# Savio: Monitoring jobs and cluster status

Use of squeue as on SCF. Also the `wwall` command shows resource usage.

```
squeue 

squeue -u paciorek

squeue -j JOB_ID

wwall -j JOB_ID
```


# Savio: Interactive jobs

You can also do work interactively.

For this, you may want to have used the -Y flag to ssh if you are running software with a GUI such as MATLAB. 

```
# ssh -Y SAVIO_USERNAME@hpc.brc.berkeley.edu
srun -A co_stat -p savio2  -N 1 -t 10:0 --pty bash
# now execute on the compute node:
module load matlab
matlab
```


# Savio: Parallel jobs

Here are some common paradigms on Savio:

 - multi-core or multi-process jobs on one node
     - `--nodes=1 --ntasks-per-node=1 --cpus-per-task=c` 
     - `--nodes=1 --ntasks-per-node=n --cpus-per-task=1` 
 - MPI jobs that use *one* CPU per task for each of *n* SLURM tasks
     - `--ntasks=n --cpus-per-task=1` 
     - `--nodes=x --ntasks-per-node=y --cpus-per-task=1` 
        - assumes that `n = x*y`
 - hybrid parallelization jobs (e.g., MPI+threading) that use *c* CPUs for each of *n* SLURM tasks
     - `--ntasks=n --cpus-per-task=c`
     - `--nodes=x --ntasks-per-node=y cpus-per-task=c` 
        - assumes that `y*c` equals the number of cores on a node and that `n = x*y` equals the total number of tasks

In general, the defaults for the various flags will be 1 so some of the flags above are not strictly needed.

For Savio, there are lots more examples of job submission scripts for different kinds of parallelization (multi-node (MPI), multi-core (openMP), hybrid, etc.) [here](http://research-it.berkeley.edu/services/high-performance-computing/running-your-jobs#Job-submission-with-specific-resource-requirements). 

Let's extend our parallel iPython example to multiple nodes (see *job-parallel-savio.sh*).

```
#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=co_stat
#
# Partition:
#SBATCH --partition=savio2
#
# Number of processors
#SBATCH --ntasks=48
#
# Wall clock limit:
#SBATCH --time=00:05:00
#
## Command(s) to run:
module load python/2.7.8 pandas ipython gcc openmpi
ipcontroller --ip='*' &
sleep 20
# srun here will start as many engines as SLURM tasks
srun ipengine &   
sleep 50  # wait until all engines have successfully started
export DATADIR=/global/scratch/paciorek
ipython parallel-analysis.py
```

# Savio: Using environment variables

The story here is the same as on the SCF. You can use the various SLURM environment variables to control your code's use of parallelization.

When you write your code, you may need to specify information about the number of cores to use. SLURM will provide a variety of variables that you can use in your code so that it adapts to the resources you have requested rather than being hard-coded. 

Here are some of the variables that may be useful: SLURM_NTASKS, SLURM_NTASKS_PER_NODE, SLURM_CPUS_PER_TASK, SLURM_NODELIST, SLURM_NNODES.

To control your code and use the resources you've requested, you can use those variables in your job submission script (as above) or read those variables into your program.

For example:

```
import os                               ## Python
int(os.environ['SLURM_NTASKS'])         ## Python

as.numeric(Sys.getenv('SLURM_NTASKS'))  ## R

str2num(getenv('SLURM_NTASKS')))        ## MATLAB
```

# Savio: Low-priority queue

Condo users have access to the broader compute resource that is limited only by the size of partitions, under the *savio_lowprio* QoS (queue). However this QoS does not get a priority as high as the general QoSs, such as *savio_normal* and *savio_debug*, or all the condo QoSs, and it is subject to preemption when all the other QoSs become busy. 

More details can be found [in the *Low Priority Jobs* section of the user guide](http://research-it.berkeley.edu/services/high-performance-computing/user-guide).

Suppose I wanted to burst beyond the Statistics condo to run on 20 nodes. I'll illustrate here with an interactive job though usually this would be for a batch job.

```
srun --account=co_stat --partition=savio2 --qos=savio_lowprio \
      --nodes=20 --time=0:10:00 --pty bash
env | grep SLURM
```

# Savio: GPU jobs

You'll need to indicate the need for one (or more) GPUs in your submission script.

Also, the GPU nodes will run jobs from multiple users at once. For each GPU that you request, you should request twice as many CPUs.

```
#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=co_stat
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# GPU:
#SBATCH --gres=gpu:1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify total number of processors twice of number of GPUs):
#SBATCH --cpus-per-task=2
#
# Wall clock limit (3 hours here):
#SBATCH --time=3:00:00
#
## Command(s) to run:

# usually need cuda loaded
module load cuda
# now run your GPU job
```

If you're logged on interactively, you can run *nvidia-smi* to see the activity on the GPU.

# Savio: Spark jobs

Savio offers Spark, but does not provide an HDFS (distributed file system) in the usual way. Instead, Spark on Savio loads data off of /scratch. I've heard some indication that this results in performance reduction but don't have any further information on this.

Here's an example job submission script (*job-spark.sh*):

```
#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=fc_paciorek
#
# Partition:
#SBATCH --partition=savio2
#
# Number of nodes
#SBATCH --nodes=8
#
# Wall clock limit (1 day here):
#SBATCH --time=1-00:00:00
#
## Command(s) to run:
module load java spark
source /global/home/groups/allhands/bin/spark_helper.sh

spark-start

spark-submit --master $SPARK_URL analysis.py 

spark-stop
```

Here's a completely uninteresting PySpark session showing how to read off of /scratch:

```
lines = sc.textFile('/global/scratch/paciorek/nielsen/*/*.tsv')
lines.count()
```

# Data transfer: SCP/SFTP

We can use the *scp* and *sftp* protocols to transfer files.

You need to use the Savio data transfer node, `dtn.brc.berkeley.edu`.

Linux/Mac:

```
# to Savio, while on your local machine
scp bayArea.csv paciorek@dtn.brc.berkeley.edu:~/.
scp bayArea.csv paciorek@dtn.brc.berkeley.edu:~/data/newName.csv
scp bayArea.csv paciorek@dtn.brc.berkeley.edu:/global/scratch/paciorek/.

# from Savio, while on your local machine
scp paciorek@dtn.brc.berkeley.edu:~/data/newName.csv ~/Desktop/.
```

If you can ssh to your local machine or want to transfer files to other systems on to which you can ssh, you can syntax like this, while logged onto Savio:

```
ssh dtn
scp ~/file.csv OTHER_USERNAME@other.domain.edu:~/data/.
```

One program you can use with Windows is *WinSCP*, and a multi-platform program for doing transfers via SFTP is *FileZilla*. After logging in, you'll see windows for the Savio filesystem and your local filesystem on your machine. You can drag files back and forth.

You can package multiple files (including directory structure) together using tar:
```
tar -cvzf files.tgz dir_to_zip 
# to untar later:
tar -xvzf files.tgz
```

# Data transfer: Globus

You can use Globus Connect to transfer data data to/from Savio (and between other resources) quickly and unattended. This is a better choice for large transfers. Here are some [instructions](http://research-it.berkeley.edu/services/high-performance-computing/using-globus-connect-savio).

Globus transfers data between *endpoints*. Possible endpoints include: Savio, your laptop or desktop, NERSC, and XSEDE, among others.

Savio's endpoint is named `ucb#brc`.

SCF's endpoint is named `UC Berkeley Statistics Department`.

If you are transferring to/from your laptop, you'll need 

- 1) Globus Connect Personal set up, 
- 2) your machine established as an endpoint and 
- 3) Globus Connect Personal actively running on your machine. At that point you can proceed as below.

To transfer files, you open Globus at [globus.org](https://globus.org) and authenticate to the endpoints you want to transfer between. You can then start a transfer and it will proceed in the background, including restarting if interrupted. 

Globus also provides a [command line interface](https://docs.globus.org/cli/using-the-cli) that will allow you to do transfers programmatically, such that a transfer could be embedded in a workflow script.


# Data transfer: Box 

Box provides **unlimited**, free, secured, and encrypted content storage of files with a maximum file size of 15 Gb to Berkeley affiliates. So it's a good option for backup and long-term storage. 

You can move files between Box and your laptop using the Box Sync app. And you can interact with Box via a web browser at [http://box.berkeley.edu](http://box.berkeley.edu).

The best way to move files between Box and Savio is [via lftp as discussed here](http://research-it.berkeley.edu/services/high-performance-computing/transferring-data-between-savio-and-your-uc-berkeley-box-account). 

Here's how you logon to box via *lftp* on Savio (assuming you've set up an external password already as described in the link above):

```
ssh SAVIO_USERNAME@dtn.brc.berkeley.edu
module load lftp
lftp ftp.box.com
set ssl-allow true
user YOUR_CALNET_ID@berkeley.edu
```

```
lpwd # on Savio
ls # on box
!ls # on Savio
mkdir workshops
cd workshops # on box
lcd savio-training-intro-2016 # on savio
put foreach-doMPI.R # savio to box
get AirlineDataAll.ffData  # box to savio; 1.4 Gb in ~ 1 minute
```

One additional command that can be quite useful is *mirror*, which lets you copy an entire directory to/from Box.

```
# to upload a directory from Savio to Box 
mirror -R mydir
# to download a directory from Box to Savio
mirror mydir .
```

Be careful, because it's fairly easy to wipe out files or directories on Box.

Finally you can set up *special purpose accounts* (Berkeley SPA) so files are owned at a project level rather than by individuals.

BRC is working (long-term) on making Globus available for transfer to/from Box, but it's not available yet.

# Data transfer: bDrive (Google Drive)

bDrive provides **unlimited**, free, secured, and encrypted content storage of files with a maximum file size of 5 Tb to Berkeley affiliates.

You can move files to and from your laptop using the Google Drive app. 

There are also some third-party tools for copying files to/from Google Drive, though I've found them to be a bit klunky. This is why we recommend using Box for workflows at this point. However, BRC is also working (short-term) on making Globus available for transfer to/from bDrive, though it's not available yet.


# How to get additional help

 - For SCF resources 
    - consult@stat.berkeley.edu 
 - For initial Savio, XSEDE, and cloud computing questions:
    - consult@stat.berkeley.edu
 - For technical issues and questions about using Savio: 
    - brc-hpc-help@berkeley.edu
 - For questions about computing resources in general, including cloud computing: 
    - brc@berkeley.edu
 - For questions about data management (including HIPAA-protected data): 
    - researchdata@berkeley.edu




