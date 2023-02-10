# MlAgility Installation Guide

The following describes how to install MlAgility.

## Installing mlagility locally

Install the `mlagility` package into your environment simply run

```
pip install -e .
```

inside the mlagility repository.

## Installing Slurm support

Slurm is an open source workload manager for clusters. If you would like to use slurm to build multiple models simultaneously, please follow the instructions below. 

### Install miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Setup your Slurm environment

Go to the mlagility folder, run the following command and wait for the Slurm job to finish:

```
sbatch src/mlagility/cli/setup_venv.sh
```

### Test it

Go to the mlagility folder and build multiple models simultaneously using Slurm.

```
benchit benchmark --all -s models/selftest/ --use-slurm --build-only --cache-dir PATH_TO_A_CACHE_DIR
```