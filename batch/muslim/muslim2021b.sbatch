#!/bin/bash
#SBATCH --job-name="mu2021b"
#SBATCH --output="slout/mus.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH -t 18:00:00

source $HOME/.bash_profile
conda activate hyde

srun --ntasks=1 python ../../tweets_muslim.py 07 2021
srun --ntasks=1 python ../../tweets_muslim.py 08 2021
srun --ntasks=1 python ../../tweets_muslim.py 09 2021
srun --ntasks=1 python ../../tweets_muslim.py 10 2021
srun --ntasks=1 python ../../tweets_muslim.py 11 2021
srun --ntasks=1 python ../../tweets_muslim.py 12 2021

wait
