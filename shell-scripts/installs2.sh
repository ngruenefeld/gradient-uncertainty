#!/bin/bash
#
#SBATCH --job-name=gruenefeld-ma-qa
#SBATCH --comment="Pip Installs"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=N.Gruenefeld@campus.lmu.de
#SBATCH --chdir=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty
#SBATCH --output=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty/slurm-outputs/slurm.%j.%N.out
#SBATCH --ntasks=1

python3 -m venv v_env
source v_env/bin/activate
pip install --upgrade --no-cache-dir pip wheel
pip install --upgrade --no-cache-dir torch
deactivate