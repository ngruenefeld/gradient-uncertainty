#!/bin/bash
#
#SBATCH --job-name=gruenefeld-test
#SBATCH --comment="Testing SLURM"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=N.Gruenefeld@campus.lmu.de
#SBATCH --chdir=/home/g/gruenefeld/ma
#SBATCH --output=/home/g/gruenefeld/ma/slurm.%j.%N.out
#SBATCH --ntasks=1

source env/bin/activate
python -u qa.py
deactivate