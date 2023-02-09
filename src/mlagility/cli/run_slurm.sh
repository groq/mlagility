#!/usr/bin/env bash
script="$1"
args="$2"
target_file_folder="$3"

# shellcheck disable=SC1090
source /net/home/"$USER"/miniconda3/etc/profile.d/conda.sh
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
cd "$target_file_folder" || exit
# shellcheck disable=SC2116,SC2046
"$script" $(echo "$args")
conda deactivate
