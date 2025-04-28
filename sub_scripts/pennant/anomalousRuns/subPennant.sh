#!/bin/bash


#SBATCH --job-name PennantAPPEKG ## name that will show up in the queue
#SBATCH --output pennant-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out

##SBATCH -A myallocation  # Allocation name
#SBATCH --nodes=2       # Total # of nodes
#SBATCH --time=00:30:00   # Total run time limit (hh:mm:ss)
#SBATCH -e pennant-e%j     # Name of stderr error file
#SBATCH -p wholenode     # Queue (partition) name
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
## Load modules
module load gcc/11.2.0
module load openmpi/4.0.6

export OMP_NUM_THREADS=8

INPUT=leblancx4-2node.pnt
EXEC=pennant

time srun ${EXEC} ${INPUT}
