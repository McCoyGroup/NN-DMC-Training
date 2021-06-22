#!/bin/bash
curdir=$PWD
cd /gscratch/ilahie/mccoygrp
. VSCode/serverlib.sh

JUPYTER_PORT=8677
JUPYTER_TIME=2:00:00
JUPYTER_NODE=gpu
JUPYTER_STARTDIR=/gscratch/ilahie/mccoygrp/nn_project
jupyter_start
cd $curdir
