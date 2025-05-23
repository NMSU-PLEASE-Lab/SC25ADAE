#!/bin/bash
#SBATCH --job-name PennantCLEAN ## name that will show up in the queue
#SBATCH --output pennant-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH -A cis240673
#SBATCH --nodes=16       # Total # of nodes
#SBATCH --time=00:90:00   # Total run time limit (hh:mm:ss)
#SBATCH -p wholenode     # Queue (partition) name
#SBATCH --exclusive

## Load modules
module load gcc/11.2.0
module load openmpi/4.0.6

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

INPUT=../../leblancx4-16node.pnt
EXEC=../../../pennant.APPEKG

echo "pennant.APPEKG jobID $SLURM_JOB_ID"
pwd
echo -e "time srun --ntasks-per-node 8 -c 1 --cpu-bind=cores ${EXEC} ${INPUT}\n"

time srun --ntasks-per-node 8 -c 1 --cpu-bind=cores ${EXEC} ${INPUT}

echo -e "\nseff stats"
seff $SLURM_JOB_ID
