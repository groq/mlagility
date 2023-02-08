# MLAgility Benchmark

This directory contains the MLAgility benchmark, which is a large collection of models that can be evaluated using the [`benchit` CLI tool](https://github.com/groq/mlagility/blob/main/docs/benchit_user_guide.md).

## Table of Contents

- [Benchmark Organization](#benchmark-organization)
- [Running the Benchmark](#running-the-benchmark)
- [Model Template](#model-template)

## Benchmark Organization

The MLAgility benchmark is made up of several corpora of models (_corpora_ is the plural of _corpus_... we had to look it up too). Each corpus is named after the online repository that the models were sourced from. Each corpus gets its own subdirectory in the `models` directory. 

The corpora in the MLAgility benchmark are:
- `diffusers`: models from the [Huggingface `diffusers` library](https://huggingface.co/docs/diffusers/index), including the models that make up Stable Diffusion.
- `graph_convolutions`: Graph Neural Network (GNN) models from a variety of publications. See the docstring on each .py file for the source.
- `popular_on_huggingface`: hundreds of the most-downloaded models from the [Huggingface models repository](https://huggingface.co/models).
- `selftest`: a small corpus with small models that can be used for testing out the MLAgility tools.
- `torch_hub`: a variety of models, including many image classification models, from the [Torch Hub repository](https://github.com/pytorch/hub).
- `torchvision`: image recognition models from the [`torchvision` library](https://pytorch.org/vision/stable/index.html).
  - _Note_: the `torchvision` library also includes many image classification models, but we excluded them to avoid overlap between our `torch_hub` and `torchvision` corpora.
- `transformers_keras`: Transformer models from the [Huggingface `transformers` library](https://huggingface.co/docs/transformers/index), expressed in the Keras framework.
- `transformers_pytorch`: Transformer models from the [Huggingface `transformers` library](https://huggingface.co/docs/transformers/index), expressed in the PyTorch framework.

## Running the Benchmark

### Prerequisites

Before running the benchmark we suggest you familiarize yourself with the [`benchit` CLI tool](https://github.com/groq/mlagility/blob/main/docs/benchit_user_guide.md) and some of the [`benchit` CLI examples](https://github.com/groq/mlagility/tree/main/examples/cli).

You must also run the following command to install all of the models' dependencies into your Python environment.

`pip install -r models/requirements.txt`

### Benchmarking Commands

Once you have fulfilled the prerequisites, you can evaluate one model from the benchmark with a command like this:

```
cd MLAGILITY_ROOT/models # MLAGILITY_ROOT is where you cloned mlagility
benchit benchmark linear.py --search-dir selftest
```

You can also evaluate an entire corpus with a command like this:
```
cd MLAGILITY_ROOT/models # MLAGILITY_ROOT is where you cloned mlagility
benchit benchmark --all --search-dir selftest
```

You can aggregate all of the benchmarking results from your `mlagility cache` into a CSV file with:

```
benchit report
```

If you want to only report on a subset of models, we recommend benchmarking those models into a specific cache directory:

```
# Benchmark the models into a specific cache directory
benchit benchmark --all --search-dir selftest --cache-dir selftest_results

# Report the results from the `selftest_results` cache
benchit report --cache-dir selftest_results
```

## Model Template

Each model in the MLAgility benchmark follows a template to keep things consistent. This template is meant to be as simple as possible. The models themselves should also be instantiated and called in a completely vendor-neutral manner.

### The model.py File

Each model in the MLAgility benchmark is hosted in a `model.py` file. This file is a Python script that must meet the following requirements:
1. Instantiate at least one model and invoke it against some inputs. The size and shape of the inputs will be used for benchmarking.
1. Provide a docstring that provides information about where the model was sourced from.

The file should be named according to the model within. For example, our file for the BERT model is named `bert.py`.

Each `model.py` can optionally include a set of [labels](#labels) and [parameters](#parameters).

### Labels

MLAgility uses labels to help organize the results data. Labels must be in the first line of the Python file and start with `# labels: `

Each label must have the format `key::value1,value2,...`

Example:

```
# labels: author::google test_group::daily,monthly
```
     
Labels are saved in your cache directory and can later be retrieved using the function `mlagility.common.labels.load_from_cache()`, which receives the `cache_dir` and `build_name` as inputs and returns the labels as a dictionary. 

### Parameters

MLAgility supports parameterizing models so that you can sweep over interesting properties such as batch size, sequence length, image size, etc. The `mlagility.parser.parse()` method parses a set of standardized parameters that we have defined and provides them to the model as it is instantiated.

For example, this code would retrieve user-defined `batch_size` and `max_seq_length` (maximum sequence length) parameters.

```
from mlagility.parser import parse
parse(["batch_size", "max_seq_length"])
```

You can pass parameters into a benchmarking run with the `--script-args` argument to `benchit`. For example, the command:

```
benchit bert.py --search-dir transformers_pytorch --script-args="--batch-size=8,--max-seq-length=128"
```

would set `batch_size=8` and `max_seq_length=128` for that benchmarking run.


The standardized set of arguments is:

- General args
    - "batch_size": Arg("batch_size", default=1, type=int),
    - "max_seq_length": Arg("max_seq_length", default=128, type=int),
    - "max_audio_seq_length": Arg("max_audio_seq_length", default=25600, type=int),
    - "height": Arg("height", default=224, type=int),
    - "num_channels": Arg("num_channels", default=3, type=int),
    - "width": Arg("width", default=224, type=int),
- Args for Graph Neural Networks
    - "k": Arg("k", default=8, type=int),
    - "alpha": Arg("alpha", default=2.2, type=float),
    - "out_channels": Arg("out_channels", default=16, type=int),
    - "num_layers": Arg("num_layers", default=8, type=int),
    - "in_channels": Arg("in_channels", default=1433, type=int),- 

### Example Model

The following example, copied from `models/transformers_keras/ctrl.py` is a good example of a well-formed `model.py` file. You can see that it has the following properties:

1. Labels in the top line of the file.
1. Docstring indicating where the model was sourced from.
1. `mlagility.parser.parse()` is used to parameterize the model.
1. The model is instantiated and invoked against a set of inputs.

```
# labels: test_group::mlagility name::ctrl author::huggingface_keras
"""
https://huggingface.co/ctrl
"""

from mlagility.parser import parse
import transformers
import tensorflow as tf

tf.random.set_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


config = transformers.AutoConfig.from_pretrained("ctrl")
model = transformers.TFAutoModel.from_config(config)
inputs = {
    "input_ids": tf.ones(shape=(batch_size, max_seq_length), dtype=tf.int32),
}

# Call model
model(**inputs)
```