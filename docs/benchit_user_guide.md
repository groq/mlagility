# MLAgility User Guide

The MLAgility Benchmarking and Tools package provides a CLI, `benchit`, and Python API, `benchit()` for benchmarking machine learning and deep learning models. This document reviews the functionality provided by MLAgility. If you are looking for repo and code organization, you can find that [here](https://github.com/groq/mlagility/blob/main/docs/code.md).

MLAgility's tools currently support the following combinations of runtimes and devices:

<span id="devices-runtimes-table">

| Device Type | Device arg | Runtime      | Runtime arg | Specific Devices                                         |
|-------------|------------|--------------|-------------|----------------------------------------------------------|
| Nvidia GPU  | nvidia     | TensorRT     | trt         | Any Nvidia GPU supported by TensorRT >= 8.5.2            |
| x86 CPU     | x86        | ONNX Runtime | ort         | Any Intel or AMD CPU supported by ONNX Runtime >= 1.13.1 |
| Groq        | groq       | GroqFlow     | groq        | GroqChip1                                                |

</span>

## Table of Contents

- [Just Benchmark It](#just-benchmark-it)
- [The benchit() API](#the-benchit-api)
- [Definitions](#definitions)
- [Devices and Runtimes](#devices-and-runtimes)
- [Additional Commands and Options](#additional-commands-and-options)

## Just Benchmark It

The simplest way to get started with MLAgility's tools is to use our `benchit` command line interface (CLI), which can take any python script that instantiates and calls PyTorch/Keras model(s) and benchmark them on any supported device and runtime.

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

Where `your_script.py` is a Python script that instantiates and executes a PyTorch/Keras model named `YourModel`. The benchmarking results are also saved to a file, `*_state.yaml`, where `*` is the `build` name (see [Build](#build)).

The `benchit` CLI performs the following steps:
1. [Analysis](#analysis): profile the Python script to identify the PyTorch/Keras models within
2. [Build](#build): call the `benchit()` [API](#the-benchit-api) to prepare each model for benchmarking
3. [Benchmark](#benchmark): call the `benchit()` [API](#the-benchit-api) on each model to gather performance statistics

_Note_: The benchmarking methodology is defined [here](#benchmark).

## The benchit() API

The [build](#build) and [benchmark](#benchmark) functionality provided by the `benchit` CLI is also available in the `benchit()` API. The main difference is that `benchit()` does not include the [Analysis](#analysis) feature of the CLI, and must be called in your Python script against a specific set of `model` and model `inputs`.

For example, the following script:

```python
from mlagility import benchit

model = YourModel()
results = model(**inputs)
perf = benchit(model, inputs)
```

Will print an output like this:

```
> Performance of YourModel on device Intel® Xeon® Platinum 8380 is:
> latency: 0.033 ms
> throughput: 21784.8 ips
```

`benchit()` also returns a `Performance` object that includes members:
 - `latency_units`: unit of time used for measuring latency, which is set to `milliseconds (ms)`.
 - `latency`: average benchmarking latency, measured in `latency_units`.
 - `throughput_units`: unit used for measuring throughput, which is set to `inferences per second (IPS)`.
 - `throughput`: average benchmarking throughput, measured in `throughput_units`.

 _Note_: The benchmarking methodology is defined [here](#benchmark).

## Definitions

MLAgility uses the following definitions throughout.

### Model

A **model** is a PyTorch (torch.nn.Module) or Keras (tf.keras.Model) object that has been instantiated in a Python script.

- Examples: BERT-Base, ResNet-50, etc.

### Device

A **device** is a piece of hardware capable of running a model.

- Examples: Nvidia A100 40GB, Intel Xeon Platinum 8380, Groq GroqChip1

### Runtime

A **runtime** is a piece of software that executes a model on a device.

- Different runtimes can produce different performance results on the same device because:
  - Runtimes often optimize the model prior to execution.
  - The runtime is responsible for orchestrating data movement, device invocation, etc.
- Examples: ONNX Runtime, TensorRT, PyTorch Eager Execution, etc.

### Analysis

**Analysis** is the process by which `benchit` inspects a Python script and identifies the PyTorch/Keras models within.

`benchit` performs analysis by running and profiling your script. When a model object (see [Model](#model) is encountered, the `benchit` CLI can inspect it to gather statistics (such as the number of parameters in the model) and/or pass it to the `benchit()` API for benchmarking.

_Note_: the `benchit` CLI runs your entire script. Please ensure that your script is safe to run, especially if you got it from the internet.

### Model Hashes

Each `model` in a `script` is identified by a unique `hash`. The `analysis` phase of `benchit` will display the `hash` for each model. The `build` phase will save exported models to into the `cache` according to the naming scheme `{script_name}_{hash}`.

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

### Build

**Build** is the process by which the `benchit()` API consumes a [model](#model) and produces ONNX files, Groq executables, and other artifacts needed for benchmarking.

We refer to this collection of artifacts as the `build directory` and store each build in the MLAgility `cache` for later use.

We leverage ONNX files because of their broad compatibility with model frameworks (PyTorch, Keras, etc.), software (ONNX Runtime, TensorRT, Groq Compiler, etc.), and devices (CPUs, GPUs, GroqChip processors, etc.). You can learn more about ONNX [here](https://onnx.ai/).

The build functionality of `benchit` includes the following steps:
1. Take a `model` object and a corresponding set of `inputs`*.
1. Check the cache for a successful build we can load. If we get a cache hit, the build is done. If no build is found, or the build in the cache is stale**, continue.
1. Pass the `model` and `inputs` to the ONNX exporter corresponding to the `model`'s framework (e.g., PyTorch models use `torch.onnx.export()`).
1. Use [ONNX Runtime](https://github.com/microsoft/onnxruntime) and [ONNX ML tools](https://github.com/onnx/onnxmltools) to optimize the model and convert it to float16, respectively.
1. [If the build's device type is `groq`] Pass the optimized float16 ONNX file to Groq Compiler and Assembler to produce a Groq executable.
1. Save the successful build to the cache for later use.

*_Note_: Each `build` corresponds to a set of static input shapes. `inputs` are passed into the `benchit()` API to provide those shapes.

**_Note_: A cached build can be stale because of any of the following changes since the last build:
* The model changed
* The shape of the inputs changed
* The arguments to `benchit` changed
* MLAgility was updated to a new, incompatible version

### Benchmark

*Benchmark* is the process by which the `benchit()` API collects performance statistics about a [model](#model). Specifically, `benchit()` takes a [build](#build) of a model and executes it on a target device using target runtime software (see [Devices and Runtimes](#devices-and-runtimes)).

By default, `benchit()` will run the model 100 times to collect the following statistics:
1. Mean Latency, in milliseconds (ms): the average time it takes the runtime/device combination to execute the model/inputs combination once. This includes the time spent invoking the device and transferring the model's inputs and outputs between host memory and the device (when applicable).
1. Throughput, in inferences per second (IPS):  the number of times the model/inputs combination can be executed on the runtime/device combination per second.
    - __Note__: `benchit()` is not aware of whether `inputs` is a single input or a batch of inputs. If your `inputs` is actually a batch of inputs, you should multiply `benchit()`'s reported IPS by the batch size.

## Devices and Runtimes

MLAgility can be used to benchmark a model across a variety of runtimes and devices, as long as the device is available and the device/runtime combination is supported by MLAgility.

### Available Devices

MLAgility supports benchmarking on both locally installed devices, as well as devices on remote machines (e.g., cloud VMs).

If you are using a remote machine, it must:
- turned on
- be available via SSH
- include the target device
- have `python>=3.8` and `docker>=20.10` installed

When you call `benchit`, the following actions are performed on your behalf:
1. Perform a `build`, which exports all models from the script to ONNX and prepares for benchmarking.
    - If the device type selected is `groq`, this step also compiles the ONNX file into a Groq executable.
1. [Remote mode only] `ssh` into the remote machine and transfer the `build`.
1. Set up the benchmarking environment by loading a container and/or setting up a Python virtual environment.
1. Run the benchmarks.
1. [Remote mode only] Transfer the results back to your local machine.

### Arguments

The following arguments are used to configure `benchit` to target a specific device and runtime:
- `--ip IP` is the ip address where the device can be located.
  - Defaults to `localhost`, indicating the device is installed on the local machine.
  - Typically, this will point to a remote VM where a target device is installed.
  - Also available as an API argument, `benchit(ip=...,)`.
- `--device TYPE` is the type of device to be used.
  - _Note_: MLAgility is flexible with respect to which specific devices can be used, as long as they meet the requirements in the [Devices and Runtimes table](#devices-runtimes-table).
    - The `benchit()` API will simply use whatever device, of the given `TYPE`, is available on the machine specified by `--ip`.
    - For example, if you specify `--device nvidia` and an `IP` that corresponds to a VM with an Nvidia A100 40GB installed, then MLAgility will use that Nvidia A100 40GB device.
  - Valid values include:
    - `x86` (default): Intel and AMD x86 CPUs.
    - `nvidia`: Nvidia GPUs.
    - `groq`: Groq GroqChip processors.
  - Also available as an API argument, `benchit(device=...,)`.
- `--runtime SW` is the runtime to be used.
  - _Note_: Each device type has its own default runtime, as indicated below.
  - Valid values include:
    - `ort`: ONNX Runtime (default for `x86` device type).
    - `trt`: Nvidia TensorRT (default for `nvidia` device type).
    - `groq`: GroqFlow (default for `groq` device type).
    - [future] `pytorch1`: PyTorch 1.x-style eager execution.
    - [future] `pytorch2`: PyTorch 2.x-style compiled graph execution.
    - [future] `ort-*`: Specific [ONNX Runtime execution providers](#https://onnxruntime.ai/docs/execution-providers/)
  - Also available as an API argument, `benchit(runtime=...,)`.

## Additional Commands and Options

`benchit` provides a variety of additional commands and options for users.

The default usage of `benchit` is to directly provide it with a python script, for example `benchit example.py --device x86`. However, `benchit` also supports the usage `benchit COMMAND`, to accomplish some additional tasks.

_Note_: Some of these tasks have to do with the MLAgility `cache`, which stores the `build directories` (see [Build](#build)).

The commands are:
- [`benchmark`](#benchmark-command) (default command): benchmark the model(s) in one or more scripts
- [`list`](#list-command): list the available builds in the cache
- [`state`](#state-command): print the [state](https://github.com/groq/groqflow/blob/main/docs/user_guide.md#stateyaml-file) of one or more builds from the cache
- [`delete`](#delete-command): delete one or more builds from the cache
- [`report`](#report-command): print a report in .csv format summarizing the results
- [`version`](#version-command): print the `benchit` version number

You can see the options available for any command by running `benchit COMMAND --help`.

### `benchmark` Command

The `benchmark` command supports the arguments from [Devices and Runtimes](#devices-and-runtimes), as well as:

- `-s SEARCH_DIR, --search-dir SEARCH_DIR` Path to a directory (defaults to the command line command line location), which serves as the search path for input scripts
  - Not available as an API argument.
- `-i INPUT_SCRIPTS [INPUT_SCRIPTS ...], --input-scripts INPUT_SCRIPTS [INPUT_SCRIPTS ...]` Name(s) of script (.py) files, within the search directory, to be built (defaults to ["all"], which uses all script files).
  - You can leverage model hashes (see [Model Hashes](#model-hashes)) at build or benchmarking time in the following manner:
    - `benchit benchmark -i example.py::hash_0` will only benchmark the model corresponding to `hash_0`.
    - You can also supply multiple hashes, for example `benchit benchmark -i example.py::hash_0,hash_1` will benchmark the models corresponding to both `hash_0` and `hash_1`.
  - Not available as an API argument.
- `--use-slurm` Execute the build(s) on Slurm instead of using local compute resources
  - Not available as an API argument.
- `--lean-cache` Delete all build artifacts except for log files after the build
- Also available as an API argument, `benchit(lean_cache=True/False, ...)`.
- `-d CACHE_DIR, --cache-dir CACHE_DIR` MLAgility build cache directory where the resulting build directories will be stored (defaults to ~/.cache/mlagility)
  - Also available as an API argument, `benchit(cache_dir=Str, ...)`.
- `--rebuild REBUILD` Sets a cache policy that decides whether to load or rebuild a cached build.
  - Takes one of the following values:
    - *Default*: `"if_needed"` will use a cached model if available, build one if it is not available, and rebuild any stale builds.
    - Set `"always"` to force `benchit` to always rebuild your model, regardless of whether it is available in the cache or not.
    - Set `"never"` to make sure `benchit` never rebuilds your model, even if it is stale. `benchit` will attempt to load any previously built model in the cache, however there is no guarantee it will be functional or correct.
  - Also available as an API argument, `benchit(rebuild=Str, ...)`.

The following options can be used to customize the analysis process (see [Analysis](#analysis)):
- _Note_: None of the following are available as API arguments.
- `--script-args SCRIPT_ARGS` Sets command line arguments for the input script. Format these as a comma-delimited string.
- `--max-depth DEPTH` Depth of sub-models to inspect within the script. Default value is 0, indicating to only analyze models at the top level of the script. Depth of 1 would indicate to analyze the first level of sub-models within the top-level models.
  - _Note_: `--max-depth` values greater than 0 are only supported for PyTorch models.

The following options are specific to Groq builds and benchmarks, and are passed into the [GroqFlow build tool](https://github.com/groq/groqflow):
- `--compiler-flags COMPILER_FLAGS [COMPILER_FLAGS ...]` Sets the groqit(compiler_flags=...) arg within the GroqFlow build tool (default behavior is to use groqit()'s default compiler flags)
  - Also available as an API argument, `benchit(compiler_flags=List, ...)`.
- `--assembler-flags ASSEMBLER_FLAGS [ASSEMBLER_FLAGS ...]` Sets the groqit(assembler_flags=...) arg within the GroqFlow build tool (default behavior is to use groqit()'s default assembler flags)
  - Also available as an API argument, `benchit(assembler_flags=List, ...)`.
- `--num-chips NUM_CHIPS` Sets the groqit(num_chips=...) arg (default behavior is to let groqit() automatically select the number of chips)
  - Also available as an API argument, `benchit(num_chips=Int, ...)`.
- `--groqview` Enables GroqView for the build(s)
  - Also available as an API argument, `benchit(groqview=True/False,)`.

Finally, you may find yourself wanting to run a subset of the benchmarking command.
- The `--analyze-only` option discovers models within the target script(s) and prints information about them, but does not perform any build or benchmarking. See [Analysis](#analysis).
  - _Note_: any build- or benchmark-specific options will be ignored, such as `--ip`, `--device`, `--groqview`, etc.
  - Not available as an API argument.
- The `--build-only` builds the models within the script(s) selected, however does not run any benchmark. See [Build](#build).
  - _Note_: any benchmark-specific options will be ignored, such as `--ip`.
  - Available as an API argument, `benchit(build_only=True/False, ...)

### `list` Command

`list` presents the following options:

- `-d CACHE_DIR, --cache-dir CACHE_DIR` Search path for builds (defaults to ~/.cache/mlagility)

_Note_: `list` is not available as an API.

### `state` Command

`state` prints out each selected the build's [`state.yaml`](https://github.com/groq/groqflow/blob/main/docs/user_guide.md#stateyaml-file) files, which contains useful information about that build. The `state` command presents the following options:

- `-d CACHE_DIR, --cache-dir CACHE_DIR` Search path for builds (defaults to ~/.cache/mlagility)
- `-b BUILD_NAMES [BUILD_NAMES ...], --build-names BUILD_NAMES [BUILD_NAMES ...]` Name(s) of the specific builds to be printed, within the cache directory
- `--all` Delete all builds in the cache directory

_Note_: `state` is not available as an API.

### `delete` Command

`delete` presents the following options:

- `-d CACHE_DIR, --cache-dir CACHE_DIR` Search path for builds (defaults to ~/.cache/mlagility)
- `-b BUILD_NAMES [BUILD_NAMES ...], --build-names BUILD_NAMES [BUILD_NAMES ...]` Name(s) of the specific builds to be deleted, within the cache directory
- `--all` Delete all builds in the cache directory

_Note_: `delete` is not available as an API.

### `report` Command

`report` presents the following options:

- `-d CACHE_DIR, --cache-dir CACHE_DIR` Search path for builds (defaults to ~/.cache/mlagility)

_Note_: `report` is not available as an API.

### `version` Command

`version` does not have any options.

_Note_: `version` is not available as an API.