#!/bin/bash
#SBATCH --job-name="res-2020a"
#SBATCH --output="res.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH -t 18:00:00

source $HOME/.bash_profile
conda activate hyde

srun --ntasks=1 python tweets.py 01 2020
srun --ntasks=1 python tweets.py 02 2020
srun --ntasks=1 python tweets.py 03 2020
srun --ntasks=1 python tweets.py 04 2020
srun --ntasks=1 python tweets.py 05 2020
srun --ntasks=1 python tweets.py 06 2020

wait
