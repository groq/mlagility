# MLAgility Benchmark

This directory contains the MLAgility benchmark, which is a large collection of models that can be evaluated using the [`benchit` CLI tool](https://github.com/groq/mlagility/blob/main/docs/benchit_user_guide.md).

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

Before running the benchmark we suggest you familiarize yourself with the [`benchit` CLI tool](https://github.com/groq/mlagility/blob/main/docs/benchit_user_guide.md) and some of the [`benchit` CLI examples](https://github.com/groq/mlagility/tree/main/examples/cli).

When you are ready, you can evaluate one model from the benchmark with a command like this:

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