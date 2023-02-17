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
- `transformers`: Transformer models from the [Huggingface `transformers` library](https://huggingface.co/docs/transformers/index).

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

If you want to only report on a subset of models, we recommend saving the benchmarking results into a specific cache directory:

```
# Save benchmark results into a specific cache directory
benchit benchmark --all --search-dir selftest --cache-dir selftest_results

# Report the results from the `selftest_results` cache
benchit report --cache-dir selftest_results
```

## Model Template

Each model in the MLAgility benchmark follows a template to keep things consistent. This template is meant to be as simple as possible. The models themselves should also be instantiated and called in a completely vendor-neutral manner.

### Input Scripts

Each model in the MLAgility benchmark is hosted in a Python script (.py file). This script must meet the following requirements:
1. Instantiate at least one model and invoke it against some inputs. The size and shape of the inputs will be used for benchmarking.
  - _Note_: `benchit` supports multiple models per script, and will benchmark all models within the script.
1. Provide a docstring that provides information about where the model was sourced from.

Each script can optionally include a set of [labels](#labels) and [parameters](#parameters). See [Example Script](#example-script) for an example of a well-formed script that instantiates one model.

_Note_: All of the scripts in `models/` are also functional on their own, without the `benchit` command. For example, if you run the command:

```
python models/transformers_pytorch/bert.py
```

this will run the PyTorch version of the Huggingface `transformers` BERT model on your local CPU device.

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
benchit bert.py --search-dir transformers_pytorch --script-args="--batch_size 8 --max_seq_length 128"
```

would set `batch_size=8` and `max_seq_length=128` for that benchmarking run.

You can also use these arguments outside of `benchit`, for example the command:

```
python models/transformers_pytorch/bert.py --batch_size 8
```

would simply run BERT with a batch size of 8 in PyTorch.


The standardized set of arguments is:

- General args
    - "batch_size": Arg("batch_size", default=1, type=int),
        - Batch size for the input to the model that will be used for benchmarking
    - "max_seq_length": Arg("max_seq_length", default=128, type=int),
        - Maximum sequence length for the model's input; also the input sequence length that will be used for benchmarking
    - "max_audio_seq_length": Arg("max_audio_seq_length", default=25600, type=int),
        - Maximum sequence length for the model's audio input; also the input sequence length that will be used for benchmarking
    - "height": Arg("height", default=224, type=int),
        - Height of the input image that will be used for benchmarking
    - "num_channels": Arg("num_channels", default=3, type=int),
        - Number of channels in the input image that will be used for benchmarking
    - "width": Arg("width", default=224, type=int),
        - Width of the input image that will be used for benchmarking
- Args for Graph Neural Networks
    - "k": Arg("k", default=8, type=int),
    - "alpha": Arg("alpha", default=2.2, type=float),
    - "out_channels": Arg("out_channels", default=16, type=int),
    - "num_layers": Arg("num_layers", default=8, type=int),
    - "in_channels": Arg("in_channels", default=1433, type=int),- 

### Example Script

The following example, copied from `models/torch_hub/alexnet.py` is a good example of a well-formed `model.py` file. You can see that it has the following properties:

1. Labels in the top line of the file.
1. Docstring indicating where the model was sourced from.
1. `mlagility.parser.parse()` is used to parameterize the model.
1. The model is instantiated and invoked against a set of inputs.

```
# labels: test_group::mlagility name::alexnet author::torch_hub
"""
https://github.com/pytorch/hub/blob/master/pytorch_vision_alexnet.md
"""

from mlagility.parser import parse
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, num_channels, width, height = parse(
    ["batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = torch.hub.load(
    "pytorch/vision:v0.13.1",
    "alexnet",
    weights=None,
)
model.eval()
inputs = {"x": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
```
