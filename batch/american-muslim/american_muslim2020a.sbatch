#!/bin/bash
#SBATCH --job-name="am2020a"
#SBATCH --output="slout/am.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH -t 18:00:00

source $HOME/.bash_profile
conda activate hyde

srun --ntasks=1 python ../../tweets_american_muslim.py 01 2020
srun --ntasks=1 python ../../tweets_american_muslim.py 02 2020
srun --ntasks=1 python ../../tweets_american_muslim.py 03 2020
srun --ntasks=1 python ../../tweets_american_muslim.py 04 2020
srun --ntasks=1 python ../../tweets_american_muslim.py 05 2020
srun --ntasks=1 python ../../tweets_american_muslim.py 06 2020

wait
