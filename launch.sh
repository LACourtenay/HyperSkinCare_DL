#!/bin/bash

# SCAYLE Supercomputacion CyL

# Instrucciones para el gestor de trabajos SLURM
# Current Working directory
#SBATCH -D .

# Partition and QoS for the job:
#SBATCH -p broadwellgpu
#SBATCH -q normal

# The name of the job:
#SBATCH --job-name="CNSVM Training"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=18
#SBATCH --gres=gpu:1

# Output
#SBATCH -o Output_Python_%j.out

#Time limit
#SBATCH --time=5-00:00:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Define paths and environment variables
module load python_3.7.4 broadwell/gcc_8.2.0 broadwell/CUDA_10.1

# Run code
python main.py
