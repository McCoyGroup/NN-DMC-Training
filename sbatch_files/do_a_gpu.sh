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

py_fl="$SLURM_JOB_NAME.py"

# JUPYTER_PORT="$2"
# if [ "$JUPYTER_PORT" == "" ]; then
#     JUPYTER_PORT=8899;
# fi

# # If you want MPI to work (you don't) set the I_like_MPI flag
# I_like_MPI=""
# if [ "$I_like_MPI" != ""]; then

# mcenv=$(mcenv -G -e)

# module load icc_19-ompi_3.1.4
# unset CC
# unset CXX

# mpirun -n 1 $mcenv --exec jupyter notebook --port=$JUPYTER_PORT --config=notebooks/jupyter_config.py
# else

# mcenv -G --exec jupyter notebook --port=$JUPYTER_PORT --config=notebooks/jupyter_config.py

# fi

mcenv -G --fulltb --script "$py_fl"
