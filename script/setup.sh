#!/bin/bash
# update conda
conda update -n base -c defaults conda
# create conda environment
conda env create -f environment.yml
# activate conda environment
echo "source activate iasi" > ~/.bashrc
