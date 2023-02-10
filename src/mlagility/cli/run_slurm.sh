#!/usr/bin/env bash
script="$1"
args="$2"
working_directory="$3"

# shellcheck disable=SC1090
conda init
conda activate tracker_slurm
export USING_SLURM="TRUE"
export HF_DATASETS_CACHE="/net/ml-cache/huggingface"
export TRANSFORMERS_CACHE="/net/ml-cache/huggingface"
export HF_HOME="/net/ml-cache/huggingface"
export XDG_CACHE_HOME="/net/ml-cache/huggingface"
export TORCH_HOME="/net/ml-cache/torch-hub"
export TORCH_HUB="/net/ml-cache/torch-hub"
export GROQFLOW_INTERNAL_FEATURES="True"
export GROQFLOW_SKIP_SDK_CHECK="True"
umask 002
python login.py --key "${HUGGINGFACE_API_KEY}"
cd "$working_directory" || exit
# shellcheck disable=SC2116,SC2046
"$script" $(echo "$args")
conda deactivate
