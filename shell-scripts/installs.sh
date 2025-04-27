#!/bin/bash
#
#SBATCH --job-name=gruenefeld-ma-qa
#SBATCH --comment="Pip Installs"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=N.Gruenefeld@campus.lmu.de
#SBATCH --chdir=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty
#SBATCH --output=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty/slurm-outputs/slurm.%j.%N.out
#SBATCH --ntasks=1

source env/bin/activate
pip install --upgrade --no-cache-dir autoawq
deactivate
git add .
git commit -m "pip installs"
git push