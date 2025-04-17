#!/bin/bash


#SBATCH --time=3-00:00:00   # job time limit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=40G   # memory
#SBATCH -J "embeddings"   # job name
#SBATCH --mail-user=gbz6qn@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/home/gbz6qn/Documents/MSDS/DS7540/project/slurm_outputs/slurm-%A_%a_out.txt
#SBATCH --error=/home/gbz6qn/Documents/MSDS/DS7540/project/slurm_outputs/slurm-%A_%a_error.txt
#SBATCH --account=ds_6050   # allocation name


# Load modules
module purge
module load gcc
module load miniforge
source activate rl

# Run script and pipe file path in
srun python embeddings.py

# Cleanup
conda deactivate