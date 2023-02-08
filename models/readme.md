# MLAgility Benchmark

This directory contains the MLAgility benchmark, which is a large collection of models that can be evaluated using the [`benchit` CLI tool](https://github.com/groq/mlagility/blob/main/docs/benchit_user_guide.md).

## Benchmark Organization

The MLAgility benchmark is made up of several corpora of models (_corpora_ is the plural of _corpus_... we had to look it up too). Each corpus is named after the online repository that the models were sourced from. Each corpus gets its own subdirectory in the `models` directory. 

The corpora in the MLAgility benchmark are:
- 

## Running the Benchmark

Before running the benchmark we suggest you familiarize yourself with the [`benchit` CLI tool](https://github.com/groq/mlagility/blob/main/docs/benchit_user_guide.md) and some of the [`benchit` CLI examples](https://github.com/groq/mlagility/tree/main/examples/cli).

When you are ready, you can evaluate one model from the benchmark with a command like this:

```
cd models/selftest
benchit linear.py
```

You can also evaluate an entire corpus with a command like this:
```
cd models/selftest
benchit benchmark --all
```