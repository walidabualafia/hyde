#!/bin/bash
#SBATCH --job-name="res-2020b"
#SBATCH --output="res.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH -t 18:00:00

source $HOME/.bash_profile
conda activate hyde

srun --ntasks=1 python tweets.py 07 2020
srun --ntasks=1 python tweets.py 08 2020
srun --ntasks=1 python tweets.py 09 2020
srun --ntasks=1 python tweets.py 10 2020
srun --ntasks=1 python tweets.py 11 2020
srun --ntasks=1 python tweets.py 12 2020

wait
