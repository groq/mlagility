# MLAgility Tools User Guide

The MLAgility Benchmarking and Tools package provides a CLI, `benchit`, and Python API for benchmarking machine learning and deep learning models. This document reviews the functionality provided by MLAgility. If you are looking for repo and code organization, you can find that [here](https://github.com/groq/mlagility/blob/main/docs/code.md).

For a hands-on learning approach, check out the [`benchit` CLI tutorials](#https://github.com/groq/mlagility/blob/main/examples/cli/readme.md).

MLAgility's tools currently support the following combinations of runtimes and devices:

<span id="devices-runtimes-table">

| Device Type | Device arg | Runtime      | Specific Devices                                         |
| ----------- | ---------- | ------------ | -------------------------------------------------------- |
| Nvidia GPU  | nvidia     | TensorRT     | Any Nvidia GPU supported by TensorRT >= 8.5.2            |
| x86 CPU     | x86        | ONNX Runtime | Any Intel or AMD CPU supported by ONNX Runtime >= 1.13.1 |
| Groq        | groq       | GroqFlow     | GroqChip1                                                |

</span>

# Table of Contents

- [Just Benchmark It](#just-benchmark-it)
- [The benchit() API](#the-benchit-api)
- [Definitions](#definitions)
- [Devices and Runtimes](#devices-and-runtimes)
- [Additional Commands and Options](#additional-commands-and-options)

# Just Benchmark It

The simplest way to get started with MLAgility's tools is to use our `benchit` command line interface (CLI), which can take any python script that instantiates and calls PyTorch model(s) and benchmark them on any supported device and runtime.

On your command line:

```
pip install mlagility
benchit your_script.py --device x86
```

Example Output:

```
> Performance of YourModel on device Intel® Xeon® Platinum 8380 is:
> latency: 0.033 ms
> throughput: 21784.8 ips
```

Where `your_script.py` is a Python script that instantiates and executes a PyTorch model named `YourModel`. The benchmarking results are also saved to a file, `*_state.yaml`, where `*` is the `build` name (see [Build](#build)).

The `benchit` CLI performs the following steps:
1. [Analysis](#analysis): profile the Python script to identify the PyTorch models within
2. [Build](#build): call the `benchit()` [API](#the-benchit-api) to prepare each model for benchmarking
3. [Benchmark](#benchmark): call the `benchit()` [API](#the-benchit-api) on each model to gather performance statistics

_Note_: The benchmarking methodology is defined [here](#benchmark). If you are looking for more detailed instructions on how to install mlagility, you can find that [here](https://github.com/groq/mlagility/blob/main/docs/install.md).

> For a detailed example, see the [CLI Hello World tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/readme.md#hello-world).

# The MLAgility API

Most of the functionality provided by the `benchit` CLI is also available in the MLAgility API:
- `mlagility.benchmark_script()` provides the same benchmarking functionality as the `benchit` CLI: it takes a script and target device, and returns performance results.
- `mlagility.benchmark_model()` provides a subset of this functionality: it takes a model and its inputs, and returns performance results.
  - The main difference is that `benchmark_model()` does not include the [Analysis](#analysis) feature, and `benchmark_script()` does.

Generally speaking, the `benchit` CLI is a command line interface for the `benchmark_script()` API, which internally calls `benchmark_model()`. You can read more about this code organization [here](https://github.com/groq/mlagility/blob/main/docs/code.md).

For example, the following script:

```python
from mlagility import benchmark_model

model = YourModel()
results = model(**inputs)
perf = benchmark_model(model, inputs)
```

Will print an output like this:

```
> Performance of YourModel on device Intel® Xeon® Platinum 8380 is:
> latency: 0.033 ms
> throughput: 21784.8 ips
```

`benchmark_model()` returns a `MeasuredPerformance` object that includes members:
 - `latency_units`: unit of time used for measuring latency, which is set to `milliseconds (ms)`.
 - `mean_latency`: average benchmarking latency, measured in `latency_units`.
 - `throughput_units`: unit used for measuring throughput, which is set to `inferences per second (IPS)`.
 - `throughput`: average benchmarking throughput, measured in `throughput_units`.

 _Note_: The benchmarking methodology is defined [here](#benchmark).

# Definitions

MLAgility uses the following definitions throughout.

## Model

A **model** is a PyTorch (torch.nn.Module) instance that has been instantiated in a Python script.

- Examples: BERT-Base, ResNet-50, etc.

## Device

A **device** is a piece of hardware capable of running a model.

- Examples: Nvidia A100 40GB, Intel Xeon Platinum 8380, Groq GroqChip1

## Runtime

A **runtime** is a piece of software that executes a model on a device.

- Different runtimes can produce different performance results on the same device because:
  - Runtimes often optimize the model prior to execution.
  - The runtime is responsible for orchestrating data movement, device invocation, etc.
- Examples: ONNX Runtime, TensorRT, PyTorch Eager Execution, etc.

## Analysis

**Analysis** is the process by which `benchmark_script()` inspects a Python script and identifies the PyTorch models within.

`benchmark_script()` performs analysis by running and profiling your script. When a model object (see [Model](#model) is encountered, it is inspected to gather statistics (such as the number of parameters in the model) and/or pass it to the `benchmark_model()` API for benchmarking.

> _Note_: the `benchit` CLI and `benchmark_script()` API both run your entire script. Please ensure that your script is safe to run, especially if you got it from the internet.

> See the [Multiple Models per Script tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/readme.md#multiple-models-per-script) for a detailed example of how analysis can discover multiple models from a single script.

## Model Hashes

Each `model` in a `script` is identified by a unique `hash`. The `analysis` phase of `benchmark_script()` will display the `hash` for each model. The `build` phase will save exported models to into the `cache` according to the naming scheme `{script_name}_{hash}`.

For example:

```
benchit example.py --analyze-only

> pytorch_model (executed 1x - 0.15s)
>        Model Type:     Pytorch (torch.nn.Module)
>        Class:          SmallModel (<class 'linear_auto.SmallModel'>)
>        Location:       linear_auto.py, line 19
>        Parameters:     55 (<0.1 MB)
>        Hash:           479b1332
```

## Build

**Build** is the process by which the `benchmark_model()` API consumes a [model](#model) and produces ONNX files, Groq executables, and other artifacts needed for benchmarking.

We refer to this collection of artifacts as the `build directory` and store each build in the MLAgility `cache` for later use.

We leverage ONNX files because of their broad compatibility with model frameworks (PyTorch, Keras, etc.), software (ONNX Runtime, TensorRT, Groq Compiler, etc.), and devices (CPUs, GPUs, GroqChip processors, etc.). You can learn more about ONNX [here](https://onnx.ai/).

The build functionality of `benchmark_model()` includes the following steps:
1. Take a `model` object and a corresponding set of `inputs`*.
1. Check the cache for a successful build we can load. If we get a cache hit, the build is done. If no build is found, or the build in the cache is stale**, continue.
1. Pass the `model` and `inputs` to the ONNX exporter corresponding to the `model`'s framework (e.g., PyTorch models use `torch.onnx.export()`).
1. Use [ONNX Runtime](https://github.com/microsoft/onnxruntime) and [ONNX ML tools](https://github.com/onnx/onnxmltools) to optimize the model and convert it to float16, respectively.
1. [If the build's device type is `groq`] Pass the optimized float16 ONNX file to Groq Compiler and Assembler to produce a Groq executable.
1. Save the successful build to the cache for later use.

> *_Note_: Each `build` corresponds to a set of static input shapes. `inputs` are passed into the `benchmark_model()` API to provide those shapes.

> **_Note_: A cached build can be stale because of any of the following changes since the last build:
> * The model changed
> * The shape of the inputs changed
> * The arguments to `benchmark_model()` changed
> * MLAgility was updated to a new, incompatible version

## Benchmark

*Benchmark* is the process by which the `benchmark_model()` API collects performance statistics about a [model](#model). Specifically, `benchmark_model()` takes a [build](#build) of a model and executes it on a target device using target runtime software (see [Devices and Runtimes](#devices-and-runtimes)).

By default, `benchmark_model()` will run the model 100 times to collect the following statistics:
1. Mean Latency, in milliseconds (ms): the average time it takes the runtime/device combination to execute the model/inputs combination once. This includes the time spent invoking the device and transferring the model's inputs and outputs between host memory and the device (when applicable).
1. Throughput, in inferences per second (IPS):  the number of times the model/inputs combination can be executed on the runtime/device combination per second.
    > - _Note_: `benchmark_model()` is not aware of whether `inputs` is a single input or a batch of inputs. If your `inputs` is actually a batch of inputs, you should multiply `benchmark_model()`'s reported IPS by the batch size.

# Devices and Runtimes

MLAgility can be used to benchmark a model across a variety of runtimes and devices, as long as the device is available and the device/runtime combination is supported by MLAgility.

## Available Devices

MLAgility supports benchmarking on both locally installed devices (including x86 CPUs / NVIDIA GPUs), as well as devices on remote machines (e.g., remote VMs).

If you are using a remote machine, it must:
- turned on
- be available via SSH
- include the target device
- have `miniconda`, `python>=3.8`, and `docker>=20.10` installed

When you call `benchit` CLI or `benchmark_model()`, the following actions are performed on your behalf:
1. Perform a `build`, which exports all models from the script to ONNX and prepares for benchmarking.
    - If the device type selected is `groq`, this step also compiles the ONNX file into a Groq executable.
1. [Remote mode only] `ssh` into the remote machine and transfer the `build`.
1. Set up the benchmarking environment by loading a container and/or setting up a conda environment.
1. Run the benchmarks.
1. [Remote mode only] Transfer the results back to your local machine.

## Arguments

The following arguments are used to configure `benchit` and the APIs to target a specific device and runtime:

### Devices

Specify a list of device types that will be used for benchmarking.

Usage:
- `benchit benchmark INPUT_SCRIPTS --devices TYPE`
  - Benchmark the model(s) in `INPUT_SCRIPTS` on a locally installed device of type `TYPE` (eg, a locally installed Nvidia device).
- `benchit benchmark INPUT_SCRIPTS --devices [TYPE ...]`
  - Benchmark the model(s) in `INPUT_SCRIPTS` across all provided device types.

Valid values of `TYPE` include:
- `x86` (default): Intel and AMD x86 CPUs.
- `nvidia`: Nvidia GPUs.
- `groq`: Groq GroqChip processors.

> _Note_: MLAgility is flexible with respect to which specific devices can be used, as long as they meet the requirements in the [Devices and Runtimes table](#devices-runtimes-table).
>  - The `benchit()` API will simply use whatever device, of the given `TYPE`, is available on the machine.
>  - For example, if you specify `--device nvidia` on a machine with an Nvidia A100 40GB installed, then MLAgility will use that Nvidia A100 40GB device.

Also available as API arguments: 
- `benchmark_script(devices=[...])`
- `benchmark_model(device=...)`.
    - _Note_: A single call to `benchmark_model()` only supports benchmarking on one device at a time, so you must call the API once per device.

> For a detailed example, see the [CLI Nvidia tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/readme.md#nvidia-benchmarking).

### Backend (future feature)

*Future feature, not yet supported.* 

Indicates whether the device is installed on the local machine or a remote machine. Device on a remote machine is accessed over SFT SSH.

Usage:
- `benchit benchmark INPUT_SCRIPTS --backend BACKEND` 

Valid values:
- Defaults to `local`, indicating the device is installed on the local machine.
- This can also be set to `remote`, indicating the target device is installed on a remote machine.

> _Note_: while `--backend remote` is implemented, and we use it for our own purposes, it has some limitations and we do not recommend using it. The limitations are:
>- Currently requires Okta SFT authentication, which not everyone will have.
>- Not covered by our automatic testing yet.

Also available as API arguments:
- `benchmark_script(backend=...)`
- `benchmark_model(backend=...)`

### Runtime (future feature)

*Future feature, not yet implemented.* 

Indicates which software runtime should be used for the benchmark (e.g., ONNX Runtime vs. TensorRT for a GPU benchmark).

Usage:
- `benchit benchmark INPUT_SCRIPTS --runtime SW`

> _Note_: We will add support for user-selected runtimes in the future, when `benchit` supports multiple runtimes per device. At the time of this writing, there is a 1:1 mapping between all supported runtimes and devices, so there is no need for the `--runtime` argument yet.

Each device type has its own default runtime, as indicated below. Valid values include:
- `ort`: ONNX Runtime (default for `x86` device type).
- `trt`: Nvidia TensorRT (default for `nvidia` device type).
- `groq`: GroqFlow (default for `groq` device type).
- [future] `pytorch1`: PyTorch 1.x-style eager execution.
- [future] `pytorch2`: PyTorch 2.x-style compiled graph execution.
- [future] `ort-*`: Specific [ONNX Runtime execution providers](#https://onnxruntime.ai/docs/
execution-providers/)

In the future this will also be available as API arguments: 
- `benchmark_script(runtime=...)`
- `benchmark_model(runtime=...)`

# Additional Commands and Options

`benchit` and the APIs provide a variety of additional commands and options for users.

The default usage of `benchit` is to directly provide it with a python script, for example `benchit example.py --device x86`. However, `benchit` also supports the usage `benchit COMMAND`, to accomplish some additional tasks.

_Note_: Some of these tasks have to do with the MLAgility `cache`, which stores the `build directories` (see [Build](#build)).

The commands are:
- [`benchmark`](#benchmark-command) (default command): benchmark the model(s) in one or more scripts
- [`cache list`](#list-command): list the available builds in the cache
- [`cache print`](#print-command): print the [state](https://github.com/groq/groqflow/blob/main/docs/user_guide.md#stateyaml-file) of a build from the cache
- [`cache delete`](#delete-command): delete one or more builds from the cache
- [`cache report`](#report-command): print a report in .csv format summarizing the results of all builds in a cache
- [`version`](#version-command): print the `benchit` version number

You can see the options available for any command by running `benchit COMMAND --help`.

## `benchmark` Command

The `benchmark` command supports the arguments from [Devices and Runtimes](#devices-and-runtimes), as well as:

### Input Scripts

Name of one or more script (.py) files to be benchmarked.

Examples: 
- `benchit models/selftest/linear.py` 
- `benchit models/selftest/linear.py models/selftest/twolayer.py` 

The `INPUT_SCRIPTS` argument is also available as an API argument:
- `benchmark_script(input_scripts=...)`

You may also use [Bash regular expressions](https://tldp.org/LDP/Bash-Beginners-Guide/html/sect_04_01.html) to locate the scripts you want to benchmark.

Examples:
- `benchit *.py`
  - Benchmark all scripts which can be found at the current working directory.
- `benchit models/*/*.py`
  - Benchmark the entire corpora of MLAgility models.

> See the [Benchmark Multiple Scripts tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/discovery.md#benchmark-multiple-scripts) for a detailed example.

You can also leverage model hashes (see [Model Hashes](#model-hashes)) to filter which models in a script will be acted on, in the following manner:
  - `benchit example.py::hash_0` will only benchmark the model corresponding to `hash_0`.
  - You can also supply multiple hashes, for example `benchit example.py::hash_0,hash_1` will benchmark the models corresponding to both `hash_0` and `hash_1`.

> _Note_: Using bash regular expressions and filtering model by hashes are mutually exclusive. To filter models by hashes, provide the full path of the Python script rather than a regular expression.

> See the [Filtering Model Hashes tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/discovery.md#filtering-model-hashes) for a detailed example.


### Use Slurm

Execute the build(s) on Slurm instead of using local compute resources.

Usage:
- `benchit benchmark INPUT_SCRIPTS --use-slurm`
  - Use Slurm to run benchit on INPUT_SCRIPTS.
- `benchit benchmark SEARCH_DIR/*.py --use-slurm`
  - Use Slurm to run benchit on all scripts in the search directory. Each script is evaluated as its on Slurm job (ie, all scripts can be evaluated in parallel on a sufficiently large Slurm cluster).

Available as an API argument:
- `benchmark_script(use_slurm=...)`

> _Note_: Requires setting up Slurm as shown [here](https://github.com/groq/mlagility/blob/main/docs/install.md).

> _Note_: while `--use-slurm` is implemented, and we use it for our own purposes, it has some limitations and we do not recommend using it. The limitations are:
>  - Currently requires Slurm to be configured the same way that it is configured at Groq, which not everyone will have.
>  - Not covered by our automatic testing yet.

### Cache Directory

`-d CACHE_DIR, --cache-dir CACHE_DIR` MLAgility build cache directory where the resulting build directories will be stored (defaults to ~/.cache/mlagility).

Also available as API arguments:
- `benchmark_script(cache_dir=...)`
- `benchmark_model(cache_dir=...)`

> See the [Cache Directory tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/cache.md#cache-directory) for a detailed example.

### Lean Cache

`--lean-cache` Delete all build artifacts except for log files after the build.

Also available as API arguments: 
- `benchmark_script(lean_cache=True/False, ...)`
- `benchmark_model(lean_cache=True/False, ...)`

> _Note_: useful for benchmarking many models, since the `build` artifacts from the models can take up a significant amount of hard drive space.

> See the [Lean Cache tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/cache.md#lean-cache) for a detailed example.

### Rebuild Policy

`--rebuild REBUILD` Sets a cache policy that decides whether to load or rebuild a cached build.

Takes one of the following values:
  - *Default*: `"if_needed"` will use a cached model if available, build one if it is not available,and rebuild any stale builds.
  - Set `"always"` to force `benchit` to always rebuild your model, regardless of whether it is available in the cache or not.
  - Set `"never"` to make sure `benchit` never rebuilds your model, even if it is stale. `benchit`will attempt to load any previously built model in the cache, however there is no guarantee it will be functional or correct.

Also available as API arguments: 
- `benchmark_script(rebuild=...)`
- `benchmark_model(rebuild=...)`

> See the [GroqFlow rebuild examples](https://github.com/groq/groqflow/tree/main/examples) to learn more.

### Sequence File

Replaces the default build sequence in `benchmark_model()` with a custom build sequence, defined in a Python script.

Usage:
- `benchit benchmark INPUT_SCRIPTS --sequence-file FILE`

This script must define a function, `get_sequence()`, that returns an instance of `groqflow.common.stage.Sequence`. See [examples/extras/example_sequence.py](https://github.com/groq/mlagility/blob/main/examples/cli/extras/example_sequence.py) for an example of a sequence file.

Also available as API arguments:
- `benchmark_script(sequence=...)`
- `benchmark_model(sequence=...)`

> _Note_: the `sequence` argument to `benchmark_script()` can be either a sequence file or a `Sequence` instance. The `sequence` argument to `benchmark_model()` must be a `Sequence` instance.

> See the [Sequence File tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/build.md#sequence-file) for a detailed example.

### Set Script Arguments

Sets command line arguments for the input script. Useful for customizing the behavior of the input script, for example sweeping parameters such as batch size. Format these as a comma-delimited string.

Usage:
- `benchit benchmark INPUT_SCRIPTS --script-args="--batch_size=8,--max_seq_len=128"`
  - This will evaluate the input script with the arguments `--batch_size=8` and `--max_seq_len=128` passed into the input script.

Also available as an API argument:
- `benchmark_script(script_args=...)`

> See the [Parameters documentation](https://github.com/groq/mlagility/blob/main/models/readme.md#parameters) for a detailed example.

### Maximum Analysis Depth

Depth of sub-models to inspect within the script. Default value is 0, indicating to only analyze models at the top level of the script. Depth of 1 would indicate to analyze the first level of sub-models within the top-level models.

Usage:
- `benchit benchmark INPUT_SCRIPTS --max-depth DEPTH`

Also available as an API argument:
- `benchmark_script(max_depth=...)`

> _Note_: `--max-depth` values greater than 0 are only supported for PyTorch models.

> See the [Maximum Analysis Depth tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/discovery.md#maximum-analysis-depth) for a detailed example.

### Analyze Only

Instruct `benchit` or `benchmark_model()` to only run the [Analysis](#analysis) phase of the `benchmark` command.

Usage:
- `benchit benchmark INPUT_SCRIPTS --analyze-only`
  - This discovers models within the input script and prints information about them, but does not perform any build or benchmarking.

> _Note_: any build- or benchmark-specific options will be ignored, such as `--backend`, `--device`, `--groqview`, etc.

Also available as an API argument: 
- `benchmark_script(analyze_only=True/False)`

> See the [Analyze Only tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/discovery.md#analyze-only) for a detailed example.

### Build Only

Instruct `benchit` or `benchmark_model()` to only run the [Analysis](#analysis) and [Build](#build) phases of the `benchmark` command.

Usage:
- `benchit benchmark INPUT_SCRIPTS --build-only`
  - This builds the models within the input script, however does not run any benchmark.

> _Note_: any benchmark-specific options will be ignored, such as `--backend`.

Also available as API arguments:
- `benchmark_script(build_only=True/False)`
- `benchmark_model(build_only=True/False)`

> See the [Build Only tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/build.md#build-only) for a detailed example.

### Groq-Specific Arguments

The following options are specific to Groq builds and benchmarks, and are passed into the [GroqFlow build tool](https://github.com/groq/groqflow). Learn more about them in the [GroqFlow user guide](https://github.com/groq/groqflow/blob/main/docs/user_guide.md).
- `--groq-compiler-flags COMPILER_FLAGS [COMPILER_FLAGS ...]` Sets the groqit(compiler_flags=...) arg within the GroqFlow build tool (default behavior is to use groqit()'s default compiler flags)
  - Also available as API arguments: `benchmark_script(groq_compiler_flags=...)`, `benchmark_model(groq_compiler_flags=...)`.
- `--groq-assembler-flags ASSEMBLER_FLAGS [ASSEMBLER_FLAGS ...]` Sets the groqit(assembler_flags=...) arg within the GroqFlow build tool (default behavior is to use groqit()'s default assembler flags)
  - Also available as API arguments: `benchmark_script(groq_assembler_flags=...)`, `benchmark_model(groq_assembler_flags=...)`.
- `--groq-num-chips NUM_CHIPS` Sets the groqit(num_chips=...) arg (default behavior is to let groqit() automatically select the number of chips)
  - Also available as API arguments: `benchmark_script(groq_num_chips=...)`, `benchmark_model(groq_num_chips=...)`.
- `--groqview` Enables GroqView for the build(s)
  - Also available as API arguments: `benchmark_script(groqview=True/False,)`, `benchmark_model(groqview=True/False,)`.


## Cache Commands

The `cache` commands help you manage the `mlagility cache` and get information about the builds and benchmarks within it.

### `cache list` Command

`benchit cache list` prints the names of all of the builds in a build cache. It presents the following options:

- `-d CACHE_DIR, --cache-dir CACHE_DIR` Search path for builds (defaults to ~/.cache/mlagility)

> _Note_: `cache list` is not available as an API.

> See the [Cache Commands tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/cache.md#cache-commands) for a detailed example.

### `cache stats` Command

`benchit cache stats` prints out the selected the build's [`state.yaml`](https://github.com/groq/groqflow/blob/main/docs/user_guide.md#stateyaml-file) file, which contains useful information about that build. The `state` command presents the following options:

- `build_name` Name of the specific build whose stats are to be printed, within the cache directory
- `-d CACHE_DIR, --cache-dir CACHE_DIR` Search path for builds (defaults to ~/.cache/mlagility)

> _Note_: `cache stats` is not available as an API.

> See the [Cache Commands tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/cache.md#cache-commands) for a detailed example.

### `cache delete` Command

`benchit cache delete` deletes one or builds from a build cache. It presents the following options:

- `build_name` Name of the specific build to be deleted, within the cache directory
- `-d CACHE_DIR, --cache-dir CACHE_DIR` Search path for builds (defaults to ~/.cache/mlagility)
- `--all` Delete all builds in the cache directory

> _Note_: `cache delete` is not available as an API.

> See the [Cache Commands tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/cache.md#cache-commands) for a detailed example.

### `cache report` Command

`benchit cache report` analyzes the state of all builds in a build cache and saves the result to a CSV file. It presents the following options:

- `-d CACHE_DIR, --cache-dir CACHE_DIR` Search path for builds (defaults to ~/.cache/mlagility)

> _Note_: `cache report` is not available as an API.

## `version` Command

`benchit version` prints the version number of the installed `mlagility` package.

`version` does not have any options.

> _Note_: `version` is not available as an API.

## Environment Variables

There are some environment variables that can control the behavior of the MLAgility tools.

### Overwrite the Cache Location

By default, the MLAgility tools will use `~/.cache/mlagility` as the MLAgility cache location. You can override this cache location with the `--cache-dir` and `cache_dir=` arguments for the CLI and APIs, respectively.

However, you may want to override cache location for future runs without setting those arguments every time. This can be accomplished with the `MLAGILITY_CACHE_DIR` environment variable. For example:

```
export MLAGILITY_CACHE_DIR=~/a_different_cache_dir
```

### Show Traceback

By default, `benchit` and `benchmark_script()` catch any exceptions during model build and benchmark and display a simple error message like `Status: Unknown benchit error: {e}`. The intention behind this behavior is to allow the MLAgility tools to iterate over many scripts and models without stopping for exceptions. 

However, you may want to see the full traceback for each exception encountered. To do so, set the `MLAGILITY_TRACEBACK` environment variable to `True`. For example:

```
export MLAGILITY_TRACEBACK=True
```

### Preserve Terminal Outputs

By default, `benchit` and `benchmark_script()` will erase the contents of the terminal in order to present a clean status update for each script and model evaluated. 

However, you may want to see everything that is being printed to the terminal. You can accomplish this by setting the `MLAGILITY_DEBUG` environment variable to `True`. For example:

```
export MLAGILITY_DEBUG=True
```
