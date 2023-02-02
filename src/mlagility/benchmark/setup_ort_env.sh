#!/usr/bin/env bash

# Create or update conda env for running CPU benchmarks using OnnxRuntime

# Disable lint warning to allow source command to use $USER
# shellcheck disable=SC1090

# Choose environment name (defaults to ort_env)
ENV_NAME=${1:-ort_env}
source /home/"$USER"/miniconda3/etc/profile.d/conda.sh

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

# Install/ upgrade onnxruntime to the latest version
python -m pip install onnxruntime --upgrade
