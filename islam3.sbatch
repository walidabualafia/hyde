#!/bin/bash
#SBATCH --job-name="research"
#SBATCH --output="islam.slurm.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL

conda activate hyde

python tweets.py 11

python tweets.py 12
