# MlAgility Installation Guide

The following describes how to install MlAgility.

## Installing mlagility

We recommend that you install [miniconda](https://docs.conda.io/en/latest/miniconda.html) like this:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create a virtual environment like this:

```
conda create -n mla python=3.8
conda activate mla
```

And then clone and install `mlagility`:

```
git clone https://github.com/groq/mlagility.git
cd mlagility
pip install -e .
```

> _Note_: If you are planning to use the `mlagility` tools with the MLAgility benchmark, Groq, or Slurm please see the corresponding sections below.

## MLAgility Benchmark Requirements

The `mlagility` package only requires the packages to run the MLAgility benchmarking tools. If you want to run the MLAgility benchmark, you will also have to install the benchmark's requirements. We 

In your `miniconda` environment:

```
pip install -r models/requirements.txt
```

## Installing Groq support

When you install `pip install` `mlagility`, `pip` will also install the `groqflow` `pip` package for you. 

However, if you want to use mlagility with your Groq SDK and GroqChip processor hardware you must also follow the steps in the [GroqFlow Install Guide](https://github.com/groq/groqflow/blob/release/0921/docs/install.md), particularly:
1. [Prerequisites](https://github.com/groq/groqflow/blob/release/0921/docs/install.md#prerequisites): install the GroqWare™ Suite.
1. [Add GroqWare Suite to Python Path](https://github.com/groq/groqflow/blob/release/0921/docs/install.md#step-3-add-groqware-suite-to-python-path)

## Installing Slurm support

Slurm is an open source workload manager for clusters. If you would like to use Slurm to build multiple models simultaneously, please follow the instructions below. Please note that MLAgility requires the Slurm nodes to have access to at least 128GB of RAM.

### Setup your Slurm environment

Ensure that your MLAgility folder and your conda installation are both inside a shared volume that can be accessed by Slurm.
Then, run the following command and wait for the Slurm job to finish:

```
sbatch --mem=128000 src/mlagility/cli/setup_venv.sh
```

### Get an API token from Huggingface.co (optional)

Some models from Huggingface.co might require the use of an API token. You can find your api token under Settings from your Hugging Face account.

To allow slurm to use your api token, simply export your token as an environment variable as shown below:


```
export HUGGINGFACE_API_KEY=<YOUR_API_KEY>
```

### Test it

Go to the mlagility folder and build multiple models simultaneously using Slurm.

```
benchit models/selftest/*.py --use-slurm --build-only --cache-dir PATH_TO_A_CACHE_DIR
```