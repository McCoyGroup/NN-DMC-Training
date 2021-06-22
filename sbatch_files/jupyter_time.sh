#!/bin/bash
#SBATCH --job-name=ryans_script
#SBATCH --time=2-18:00:00
#SBATCH --mem=120GB
#SBATCH --account=stf 
#SBATCH --gres=gpu:P100:1
#SBATCH --partition=stf-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --chdir=.

module load singularity
. /gscratch/ilahie/mccoygrp/mcenv.sh
MCENV_IMAGE=/gscratch/ilahie/mccoygrp/mcenv.sif
MCENV_PACKAGES_PATH=/gscratch/ilahie/mccoygrp/nn_project/packages

JUPYTER_PORT=8870;

mcenv -G --exec jupyter notebook --port=$JUPYTER_PORT --config=notebooks/jupyter_config.py
