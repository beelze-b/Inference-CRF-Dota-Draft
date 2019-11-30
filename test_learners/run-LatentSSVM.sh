#!/bin/bash

#SBATCH --job-name=LatentSSVM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=bak438@nyu.edu
#SBATCH --output=slurm-%j.out

module purge

. /scratch/bak438/anaconda3/bin/activate
conda activate dota_project

python python_scripts/LatentSSVM.py 
