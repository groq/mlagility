#!/usr/bin/env bash
#SBATCH --mem=64000
#SBATCH --cpus-per-task 1

## Disable lint warning to allow source command to use $USER
# shellcheck disable=SC1090

## Create venv and install most packages
source /net/home/"$USER"/miniconda3/etc/profile.d/conda.sh
conda create -n groqit_slurm python=3.8
conda activate groqit_slurm
python -m pip install --upgrade pip
cd /net/home/"$USER"/Groq/sales/groqit || exit
pip install -e .
