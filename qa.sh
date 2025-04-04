#!/bin/bash
#
#SBATCH --job-name=gruenefeld-ma-qa
#SBATCH --comment="Running the QA script"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=N.Gruenefeld@campus.lmu.de
#SBATCH --chdir=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty
#SBATCH --output=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty/slurm-outputs/slurm.%j.%N.out
#SBATCH --ntasks=1

source env/bin/activate
python -um scripts.qa
deactivate
git add .
git commit -m "QA script run"
git push