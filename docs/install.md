# MlAgility Installation Guide

The following describes how to install MlAgility.

## Installing mlagility locally

Install the `mlagility` package into your environment simply run

```
pip install -e .
```

inside the mlagility repository.

## Installing Slurm support

### Install miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Setup your slurm environment

Go to the mlagility, run the following command and wait for the slurm job to finish:

```
sbatch src/mlagility/cli/setup_venv.sh
```
