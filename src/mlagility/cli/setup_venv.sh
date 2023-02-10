#!/usr/bin/env bash
#SBATCH --mem=64000
#SBATCH --cpus-per-task 1

# Note: please install miniconda before running this script

# Initialize conda
source "$(dirname $(dirname $(which conda)))"/etc/profile.d/conda.sh

# Parse arguments
MLAGILITY_PATH=${1:-"$PWD"}
ENV_NAME=${2:-tracker_slurm}

# Create environment (if it doen't exist)
export CONDA_ALWAYS_YES="true"
if { conda env list | grep "$ENV_NAME "; } >/dev/null 2>&1; then
    echo "$ENV_NAME already exists - Not creating it from scratch"

else
    echo "Creating $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.8
fi
unset CONDA_ALWAYS_YES

# Activate environment and upgrade pip
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# Install mlagility and model requirements
cd "$MLAGILITY_PATH" || exit
pip install -e .
cd models || exit
pip install -r requirements.txt