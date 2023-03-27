#!/bin/bash
#SBATCH --job-name="res-2021a"
#SBATCH --output="res.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH -t 18:00:00

source $HOME/.bash_profile
conda activate hyde

srun --ntasks=1 python tweets.py 01 2021
srun --ntasks=1 python tweets.py 02 2021
srun --ntasks=1 python tweets.py 03 2021
srun --ntasks=1 python tweets.py 04 2021
srun --ntasks=1 python tweets.py 05 2021
srun --ntasks=1 python tweets.py 06 2021

wait
