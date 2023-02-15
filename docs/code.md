# MLAgility Code Structure

## Repo Organization

The MLAgility repository has a few major top-level directories:
- `docs`: documentation for the entire project.
- `examples`: example scripts for use with the MLAgility benchmarking tools.
  - `examples/api`: examples scripts that invoke the benchmarking API to get the performance of models.
  - `examples/cli`: example scripts that can be fed as input into the `benchit` CLI. These scripts each have a docstring that recommends one or more `benchit` CLI commands to try out.
- `models`: the corpora of models that makes up the MLAgility benchmark (see [the models readme](https://github.com/groq/mlagility/blob/main/models/readme.md)).
  - Each subdirectory under `models` represents a corpus of models pulled from somewhere on the internet. For example, `models/torch_hub` is a corpus of models from [Torch Hub](https://github.com/pytorch/hub).
- `src/mlagility`: source code for the MLAgility benchmarking tools (see [Benchmarking Tools](#benchmarking-tools) for a description of how the code is used).
  - `src/mlagility/analysis`: functions for profiling a model script, discovering model instances, and invoking `benchmark_model()` on those instances.
  - `src/mlagility/api`: implements the benchmarking APIs.
  - `src/mlagility/cli`: implements the `benchit` CLI.
  - `src/mlagility/common`: functions common to the other modules.
  - `src/mlagility/version.py`: defines the package version number.
- `test`: tests for the MLAgility benchmarking tools.
  - `test/analysis.py`: tests focusing on the analysis of model scripts.
  - `test/cli.py`: tests focusing on top-level CLI features.

## Benchmarking Tools

MLAgility provides two main tools, the `benchit` CLI and benchmarking APIs. Instructions for how to use these tools are documented in the [Tools User Guide](https://github.com/groq/mlagility/blob/main/docs/tools_user_guide.md), while this section is about how the source code is invoked to implement the tools. All of the code below is located under `src/mlagility/`.

1. The `benchit` CLI is the comprehensive frontend that wraps all the other code. It is implemented in [cli/cli.py](https://github.com/groq/mlagility/blob/main/src/mlagility/cli/cli.py).
1. The default command for `benchit` CLI runs the `benchmark_script()` API, which is implemented in [api/script_api.py](https://github.com/groq/mlagility/blob/main/src/mlagility/api/script_api.py).
    - Other CLI commands are also implemented in `cli/`, for example the `report` command is implemented in `cli/report.py`.
1. The `benchmark_script()` API takes in a set of scripts, each of which should invoke at least one model instance, to evaluate and passes each into the `evaluate_script()` function for analysis, which is implemented in [analysis/analysis.py](https://github.com/groq/mlagility/blob/main/src/mlagility/analysis/analysis.py).
1. `evaluate_script()` uses a profiler to discover the model instances in the script, and passes each into the `benchmark_model()` API, which is defined in [api/model_api.py](https://github.com/groq/mlagility/blob/main/src/mlagility/api/model_api.py).
1. The `benchmark_model()` API prepares the model for benchmarking (e.g., exporting and optimizing an ONNX file), which creates an instance of a `*Model` class, where `*` can be CPU, GPU, etc. The `*Model` classes are defined in [api/ortmodel.py](https://github.com/groq/mlagility/blob/main/src/mlagility/api/ortmodel.py), [api/trtmodel.py](https://github.com/groq/mlagility/blob/main/src/mlagility/api/trtmodel.py), etc.
1. The `*Model` classes provide a `.benchmark()` method that benchmarks the model on the device and returns an instance of the `MeasuredPerformance` class, which includes the performance statistics acquired during benchmarking.
1. `benchmark_model()` and the `*Model` classes are built on top of the [GroqFlow](https://github.com/groq/groqflow) framework.