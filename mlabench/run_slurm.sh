#!/usr/bin/env bash
script="$1"
args="$2"

# Add any data caches used by third party model hubs (huggingface, torch hub, etc.)
export HF_DATASETS_CACHE="/net/ml-cache/huggingface"
export TRANSFORMERS_CACHE="/net/ml-cache/huggingface"
export HF_HOME="/net/ml-cache/huggingface"
export XDG_CACHE_HOME="/net/ml-cache/huggingface"
export TORCH_HOME="/net/ml-cache/torch-hub"

# shellcheck source=/dev/null
source /net/home/"$USER"/miniconda3/etc/profile.d/conda.sh
conda activate groqit_slurm
export USING_SLURM="TRUE"
# shellcheck disable=SC2046
# shellcheck disable=SC2116
"$script" $(echo "$args")
conda deactivate
