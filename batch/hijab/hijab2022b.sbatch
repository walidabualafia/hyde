#!/bin/bash
#SBATCH --job-name="hij2022b"
#SBATCH --output="slout/hij.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH -t 18:00:00

source $HOME/.bash_profile
conda activate hyde

srun --ntasks=1 python ../../tweets_hijab.py 07 2022
srun --ntasks=1 python ../../tweets_hijab.py 08 2022
srun --ntasks=1 python ../../tweets_hijab.py 09 2022
srun --ntasks=1 python ../../tweets_hijab.py 10 2022
srun --ntasks=1 python ../../tweets_hijab.py 11 2022
srun --ntasks=1 python ../../tweets_hijab.py 12 2022

wait
