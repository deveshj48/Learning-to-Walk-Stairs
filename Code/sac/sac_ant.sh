#!/bin/bash
#SBATCH -p bigbatch
#SBATCH -J sac_ant
#SBATCH -o train.log
#SBATCH -e train.err
#SBATCH -N 1
#SBATCH -t 06:00:00

python run_sac.py


