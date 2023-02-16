# Learning the `benchit` CLI

This document is a tutorial for exploring the different features of the `benchit` command line interface (CLI). You can learn the details of those features in the [Tools User Guide](https://github.com/groq/mlagility/blob/main/docs/tools_user_guide.md) and learn about their implementation in the [Code Organization Guide](https://github.com/groq/mlagility/blob/main/docs/code.md).

We've created this tutorial document because `benchit` is a CLI that benchmarks the contents of `.py` scripts. So all of the `.py` scripts in the `examples/cli/scripts` directory are meant to be fed into `benchit` to demonstrate some specific functionality.

Once you've familiarized yourself with these features, head over to the [`models` directory](https://github.com/groq/mlagility/tree/main/models) to learn how to use `benchit` with real world machine learning models.

# Table of Contents

TODO

# Tutorials

All of the tutorials assume that your command line is in the same location as this readme file (`examples/cli`).

## Hello World

We can perform a basic invocation of `benchit` to benchmark a PyTorch model by simply running the following command:

```
benchit scripts/hello_world.py
```

That commands `benchit` benchmark `hello_world.py` on your CPU. Specifically, `benchit` takes the following actions:
1. Pass `scripts/hello_world.py` as the `input_script` to the `benchmark` command of `benchit`.
  - _Note_: `benchit <.py file>` is a shorthand for `benchit benchmark <.py file>`.
1. Run `hello_world.py` against a profiler and look for machine learning models.
1. Discover the `pytorch_model` instance of class `SmallModel`, which is a PyTorch model, and print some statistics about it.
1. Export `pytorch_model` to an ONNX file, optimize that ONNX file, and convert it to the `float16` data type.
1. Measure the performance of the ONNX file on your x86 CPU and report the `mean latency` and `throughput`.

The result looks like this:

```
Models discovered during profiling:

hello_world.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'hello_world.SmallModel'>)
                Location:       /home/jfowers/mlagility/examples/cli/hello_world.py, line 29
                Parameters:     55 (<0.1 MB)
                Hash:           479b1332
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.001   milliseconds (ms)
                                Throughput:     185964.8        inferences per second (IPS)

pytorch_outputs: tensor([-0.1675,  0.1548, -0.1627,  0.0067,  0.3353], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete. Use the 'report' command to get a .csv file that summarizes results across all builds in the cache.
```

You can see on the `Status:` line that the `pytorch_model` was benchmarked on a `Intel(R) Xeon(R) CPU @ 2.20GHz` x86 device and that the mean latency and throughput are both reported.

<!--

TODO: polish up and uncomment once #116 closes

## Keras

The script `scripts/keras.py` is similar to `hello_world.py`, except that it demonstrates that `benchit` can be used with TensorFlow Keras models. 

Run this command:

```
benchit scripts/keras.py
```

To get a result like this:

```
Models discovered during profiling:

keras.py:
        trackable (executed 1x)
                Model Type:     Keras (tf.keras.Model)
                Class:          SmallKerasModel (<class 'keras.SmallKerasModel'>)
                Location:       /net/home/jfowers/miniconda3/envs/mla/lib/python3.8/site-packages/tensorflow/python/trackable/data_structures.py, line 163
                Parameters:     55 (<0.1 MB)
                Hash:           5a591a29
                Status:         Unknown benchit error: Error: Failure to run model using onnxruntime - 

keras_outputs: [[0.         0.43556815 0.         0.         1.0099151 ]]

Woohoo! The 'benchmark' command is complete. Use the 'report' command to get a .csv file that summarizes results across all builds in the cache.
```

-->

## Nvidia Benchmarking

By default, `benchit` uses x86 CPUs for benchmarking, however benchmarking on Nvidia GPUs is also supported using the `--device` argument.

If you have an Nvidia GPU installed on your machine, you can benchmark `hello_world.py` by running the following command:

```
benchit scripts/hello_world.py --device nvidia
```

To get a result like this:

```
Models discovered during profiling:

hello_world.py:
	pytorch_model (executed 1x)
		Model Type:	Pytorch (torch.nn.Module)
		Class:		SmallModel (<class 'hello_world.SmallModel'>)
		Location:	/home/jfowers/mlagility/examples/cli/hello_world.py, line 29
		Parameters:	55 (<0.1 MB)
		Hash:		479b1332
		Status:		Model successfully benchmarked on NVIDIA A100-SXM4-40GB
				Mean Latency:	0.027	milliseconds (ms)
				Throughput:	21920.5	inferences per second (IPS)

pytorch_outputs: tensor([-0.1675,  0.1548, -0.1627,  0.0067,  0.3353], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete. Use the 'report' command to get a .csv file that summarizes results across all builds in the cache.
```

You can see that the device mentioned in the status is a `NVIDIA A100-SXM4-40GB`.

## Multiple Models per Script

The MLAgility tools will benchmark all models discovered in the input script. We can demonstrate this with the `two_models.py` script.

Run the following command:

```
benchit scripts/two_models.py
```

To get a result like:

```
Models discovered during profiling:

two_models.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'two_models.SmallModel'>)
                Location:       /home/jfowers/mlagility/examples/cli/scripts/two_models.py, line 32
                Parameters:     55 (<0.1 MB)
                Hash:           479b1332
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     640717.1        inferences per second (IPS)

        another_pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'two_models.SmallModel'>)
                Location:       /home/jfowers/mlagility/examples/cli/scripts/two_models.py, line 40
                Parameters:     510 (<0.1 MB)
                Hash:           215ca1e3
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     642021.1        inferences per second (IPS)

pytorch_outputs: tensor([ 0.3628,  0.0489,  0.2952,  0.0021, -0.0161], grad_fn=<AddBackward0>)
more_pytorch_outputs: tensor([-0.1198, -0.5344, -0.1920, -0.1565,  0.2279,  0.6915,  0.8540, -0.2481,
         0.0616, -0.4501], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete. Use the 'report' command to get a .csv file that summarizes results across all builds in the cache.
```

You can see that both model instances in `two_models.py`, `pytorch_model` and `another_pytorch_model`, are both discovered and benchmarked.

## Model Hashes

When you ran the example from the [Multiple Models per Script](#multiple-models-per-script) tutorial, you saw that `benchit` discovered, built, and benchmarked two models. What if you only wanted to build and benchmark one of the models?

You can leverage the model hashes feature of `benchit` to filter which models are acted on. You can see in the result from [Multiple Models per Script](#multiple-models-per-script) that the two models, `pytorch_model` and `another_pytorch_model`, have hashes `479b1332` and `215ca1e3`, respectively.

If you wanted to only build and benchmark `another_pytorch_model`, you could use this command, which filters `two_models.py` with the hash `215ca1e3`:

```
benchit benchmark scripts/two_models.py::215ca1e3
```

That would produce a result like:

```
Models discovered during profiling:

two_models.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'two_models.SmallModel'>)
                Location:       /home/jfowers/mlagility/examples/cli/scripts/two_models.py, line 32
                Parameters:     55 (<0.1 MB)
                Hash:           479b1332

        another_pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'two_models.SmallModel'>)
                Location:       /home/jfowers/mlagility/examples/cli/scripts/two_models.py, line 40
                Parameters:     510 (<0.1 MB)
                Hash:           215ca1e3
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     499272.2        inferences per second (IPS)

pytorch_outputs: tensor([ 0.3628,  0.0489,  0.2952,  0.0021, -0.0161], grad_fn=<AddBackward0>)
more_pytorch_outputs: tensor([-0.1198, -0.5344, -0.1920, -0.1565,  0.2279,  0.6915,  0.8540, -0.2481,
         0.0616, -0.4501], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete. Use the 'report' command to get a .csv file that summarizes results across all builds in the cache.
```

You can see that both models are discovered, but only `another_pytorch_model` was built and benchmarked.

> See the [Input Script documentation](https://github.com/groq/mlagility/blob/main/tools_user_guide.md#input-script) for more details.

## Search Directory

The `--search-dir` argument to `benchit` (shorthand: `-s`) is useful for pointing `benchit` to look for the input script in a directory other than the current command line location.

For example, the command:

```
benchit benchmark scripts/hello_world.py
```

is equivalent to the command:

```
benchit benchmark hello_world.py --search-dir scripts
```

The `--search-dir` argument is not very exciting on its own, however you can see the [Benchmark All](#benchmark-all) tutorial to see how it is powerful when combined with the `--all` argument.

> See the [Search Directory documentation](https://github.com/groq/mlagility/blob/main/tools_user_guide.md#search-directory) for more details.

## Benchmark All Scripts

If you want to benchmark an entire corpus of models, but you don't want to call `benchit` individually on each model, then the `--all` argument is for you. `--all` will instruct `benchit` it iterate over every script in the search directory.

For example, the command:

```
benchit benchmark --all --search-dir scripts
```

Will iterate over every model in every script in the `scripts` directory, producing a result like this:

```

Models discovered during profiling:

hello_world.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'hello_world.SmallModel'>)
                Location:       /home/jfowers/mlagility/examples/cli/scripts/hello_world.py, line 29
                Parameters:     55 (<0.1 MB)
                Hash:           479b1332
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     657792.2        inferences per second (IPS)

two_models.py:
        another_pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'two_models.SmallModel'>)
                Location:       /home/jfowers/mlagility/examples/cli/scripts/two_models.py, line 40
                Parameters:     510 (<0.1 MB)
                Hash:           215ca1e3
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     509528.6        inferences per second (IPS)

max_depth.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          TwoLayerModel (<class 'max_depth.TwoLayerModel'>)
                Location:       /home/jfowers/mlagility/examples/cli/scripts/max_depth.py, line 41
                Parameters:     85 (<0.1 MB)
                Hash:           80b93950
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     693955.3        inferences per second (IPS)

pytorch_outputs: tensor([-0.1675,  0.1548, -0.1627,  0.0067,  0.3353], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete. Use the 'report' command to get a .csv file that summarizes results across all builds in the cache.
```

You can see that `hello_world.py`, `two_models.py`, and `max_depth.py` are all evaluated.

> See the [Benchmark All Scripts documentation](https://github.com/groq/mlagility/blob/main/tools_user_guide.md#benchmark-all-scripts) for more details.

## Cache Directory

By default, MLAgility tools use `~/.cache/mlagility/` as the location for the MLAgility Cache (see the [Build documentation](https://github.com/groq/mlagility/blob/main/tools_user_guide.md#build) for more details).

However, you might want to set the cache location for any number of reasons. For example, you might want to keep the results from benchmarking one corpus of models separate from the results from another corpus.

You can try this out with the following command:

```
benchit benchmark scripts/hello_world.py --cache-dir tmp_cache
```

When that command completes, you can use the `ls` command to see that `tmp_cache` has been created at your command line location. 

See the [Cache Commands](#cache-commands) tutorial to see what you can do with the cache.

> See the [Cache Directory documentation](https://github.com/groq/mlagility/blob/main/tools_user_guide.md#cache-directory) for more details.

## Cache Commands

This tutorial assumes you have completed the [Cache Directory](#cache-directory) and [Benchmark All Scripts](#benchmark-all-scripts) tutorials, and that the `tmp_cache` directory exists at your command line location.

You can use the `cache list` command to see what builds are available in your cache:

```
benchit cache list
```

Which produces a result like:

```
Info: Builds available in cache ~/.cache/mlagility:
hello_world_479b1332     max_depth_80b93950       two_models_479b1332      
        labels                   two_models_215ca1e3 
```

> _Note_: `labels` is not a build, it is a helper directory to store labeling data.

You can learn more about a build with the `cache stats` command:

```
benchit cache stats hello_world_479b1332
```

Which will print out a lot of statistics about the build, like:

```
Info: The state of build hello_world_479b1332 in cache ~/.cache/mlagility is:
build_status: successful_build
cache_dir: /home/jfowers/.cache/mlagility
config:
  assembler_flags:
  - --ifetch-from-self
  - --ifetch-slice-ordering=round-robin
  build_name: hello_world_479b1332
  compiler_flags: []
  groqcard: A1.4
  groqview: false
  num_chips: null
  sequence:
  - export_pytorch
  - optimize_onnx
  - fp16_conversion
  - set_success
...
```

You can also delete a build from a cache with the `cache delete` command. Be careful, this permanently deletes the build!

For example, you could run the commands:

```
benchit cache delete max_depth_80b93950
benchit cache list
```

And you would see that the cache no longer includes the build for `max_depth`:

```
Info: Builds available in cache ~/.cache/mlagility:
hello_world_479b1332     labels                   two_models_215ca1e3      two_models_479b1332 
```

Finally, the `cache` commands all take a `--cache-dir` that allows them to operate on a specific cache directory (see the [Cache Directory tutorial](#cache-directory) for more details).

For example, you can run this command:

```
benchit cache list --cache-dir tmp_cache
```

Which will produce this result, if you did the [Cache Directory tutorial](#cache-directory):

```
Info: Builds available in cache tmp_cache:
hello_world_479b1332     labels  
```

> See the [Cache Commands documentation](https://github.com/groq/mlagility/blob/main/tools_user_guide.md#cache-commands) for more details.

## Lean Cache

As you progress with MLAgility you may notice that the cache directory can take up a lot of space on your hard disk because it produces a lot of ONNX files and other build artifacts. 

We provide the `--lean-cache` option to help in the situation where you want to collect benchmark results, but you don't care about keeping the build artifacts around.

First, do the [Cache Directory tutorial](#cache-directory) so that we have a nice convenient cache directory to look at.

Next, run the command:

```
ls -shl tmp_cache/hello_world_479b1332
```

to see the contents of the cache directory, along with their file sizes:

```
total 32K
4.0K drwxr-xr-x 2 jfowers groq 4.0K Feb 16 08:14 compile
4.0K -rw-r--r-- 1 jfowers groq 2.0K Feb 16 08:14 hello_world_479b1332_state.yaml
4.0K -rw-r--r-- 1 jfowers groq  396 Feb 16 08:14 inputs_original.npy
4.0K -rw-r--r-- 1 jfowers groq   84 Feb 16 08:14 log_export_pytorch.txt
4.0K -rw-r--r-- 1 jfowers groq   71 Feb 16 08:14 log_fp16_conversion.txt
4.0K -rw-r--r-- 1 jfowers groq   63 Feb 16 08:14 log_optimize_onnx.txt
   0 -rw-r--r-- 1 jfowers groq    0 Feb 16 08:14 log_set_success.txt
4.0K drwxr-xr-x 2 jfowers groq 4.0K Feb 16 08:14 onnx
4.0K drwxr-xr-x 3 jfowers groq 4.0K Feb 16 08:14 x86_benchmark
```

These file sizes aren't too bad because the `pytorch_model` in the [Cache Directory tutorial](#cache-directory) isn't very large. But imagine if you were using GPT-J 6.7B instead, there would be tens of gigabytes of data left on your disk.

Now run the following command to repeat the [Cache Directory tutorial](#cache-directory) in lean cache mode:

```
benchit benchmark scripts/hello_world.py --cache-dir tmp_cache --lean-cache
```

And then inspect the build directory again:

```
ls -shl tmp_cache/hello_world_479b1332
```

To see that the `onnx` and `x86_benchmark` directories are gone, thereby saving disk space:

```
total 20K
4.0K drwxr-xr-x 2 jfowers groq 4.0K Feb 16 08:14 compile
4.0K -rw-r--r-- 1 jfowers groq 2.0K Feb 16 08:14 hello_world_479b1332_state.yaml
4.0K -rw-r--r-- 1 jfowers groq   84 Feb 16 08:14 log_export_pytorch.txt
4.0K -rw-r--r-- 1 jfowers groq   71 Feb 16 08:14 log_fp16_conversion.txt
4.0K -rw-r--r-- 1 jfowers groq   63 Feb 16 08:14 log_optimize_onnx.txt
   0 -rw-r--r-- 1 jfowers groq    0 Feb 16 08:14 log_set_success.txt
```

> See the [Lean Cache documentation](https://github.com/groq/mlagility/blob/main/tools_user_guide.md#lean-cache) for more details.

## Sequence File

You can customize the behavior of the [Build](https://github.com/groq/mlagility/blob/main/tools_user_guide.md#build) stage of `benchit` by creating a custom `Sequence`.

A `Sequence` tells the `benchmark_model()` API within `benchit` how to `build` a model to prepare it for benchmarking.

The default `Sequence` for CPU and GPU benchmarking performs the following build steps:
1. Export the model to an ONNX file
1. Use ONNX Runtime to optimize the ONNX file
1. Use ONNX ML Tools to the convert the optimized ONNX file to float16
1. Set the `build_status=successful_build` property

You can see this if you already did the [Hello World tutorial](#hello-world) by running the command:

```
benchit cache stats hello_world_479b1332
```

If you inspect the `sequence` field of the output, you will see:

```
  sequence:
  - export_pytorch
  - optimize_onnx
  - fp16_conversion
  - set_success
```

However, you might want a different `Sequence`. Let's say you just want to export the ONNX file, without optimizing it or converting it. Such a `Sequence` is defined in `extras/example_sequence.py`.

You can try this out with the command:

```
benchit benchmark scripts/hello_world.py --sequence-file extras/example_sequence.py --build-only
```

If we then repeat the `benchit cache stats hello_world_479b1332` we will see that the `sequence` field has changed:

```
  sequence:
  - export_pytorch
  - set_success
```

> See the [Sequence File documentation](https://github.com/groq/mlagility/blob/main/tools_user_guide.md#sequence-file) for more details.

### Maximum Analysis Depth

PyTorch and Keras models (eg, `torch.nn.Module`) are often built out of a collection of smaller instances. For example, a PyTorch multilayer perceptron (MLP) model may be built out of many `torch.nn.Linear` modules.

Sometimes you will be interested to analyze or benchmark those sub-modules, which is where the `--max-depth` argument comes in.

For example, if you run this command:

```
benchit benchmark scripts/max_depth.py
```

You will get a result that looks very similar to the [Hello World tutorial](#hello-world) tutorial. However, if you peek into `max_depth.py`, you can see that there are two `torch.nn.Linear` modules that make up the top-level model.

You can analyze and benchmark those `torch.nn.Linear` modules with this command:

```
benchit benchmark scripts/max_depth.py --max-depth 1
```

You get a result like:

```
Models discovered during profiling:

max_depth.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          TwoLayerModel (<class 'max_depth.TwoLayerModel'>)
                Location:       /home/jfowers/mlagility/examples/cli/scripts/max_depth.py, line 41
                Parameters:     85 (<0.1 MB)
                Hash:           80b93950
                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                Mean Latency:   0.000   milliseconds (ms)
                                Throughput:     533884.4        inferences per second (IPS)

max_depth.py:
                        fc (executed 2x)
                                Model Type:     Pytorch (torch.nn.Module)
                                Class:          Linear (<class 'torch.nn.modules.linear.Linear'>)
                                Parameters:     55 (<0.1 MB)
                                Hash:           6d5eb4ee
                                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                                Mean Latency:   0.000   milliseconds (ms)
                                                Throughput:     809701.4        inferences per second (IPS)

                        fc2 (executed 2x)
                                Model Type:     Pytorch (torch.nn.Module)
                                Class:          Linear (<class 'torch.nn.modules.linear.Linear'>)
                                Parameters:     30 (<0.1 MB)
                                Hash:           d4b2ffa7
                                Status:         Model successfully benchmarked on Intel(R) Xeon(R) CPU @ 2.20GHz
                                                Mean Latency:   0.000   milliseconds (ms)
                                                Throughput:     677945.2        inferences per second (IPS)
```

You can see that the two instances of `torch.nn.Linear`, `fc` and `fc2`, are benchmarked in addition to the top-level model, `pytorch_model`.

> See the [Max Depth documentation](https://github.com/groq/mlagility/blob/main/tools_user_guide.md#maximum-analysis-depth) for more details.

## Analyze Only

`benchit` provides the `--analyze-only` argument for when you want to analyze the models in a script, without actually building or benchmarking them.

You can try it out with this command:

```
benchit benchmark scripts/hello_world.py --analyze-only
```

Which gives a result like:

```
Models discovered during profiling:

hello_world.py:
        pytorch_model (executed 1x - 0.00s)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'hello_world.SmallModel'>)
                Location:       /home/jfowers/mlagility/examples/cli/scripts/hello_world.py, line 29
                Parameters:     55 (<0.1 MB)
                Hash:           479b1332

pytorch_outputs: tensor([-0.1675,  0.1548, -0.1627,  0.0067,  0.3353], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete. Use the 'report' command to get a .csv file that summarizes results across all builds in the cache.
```

You can see that the model is discovered, and some stats are printed, but no build or benchmark took place.

> See the [Analyze Only documentation](https://github.com/groq/mlagility/blob/main/tools_user_guide.md#analyze-only) for more details.

## Build Only

`benchit` provides the `--build-only` argument for when you want to analyze and build the models in a script, without actually benchmarking them.

You can try it out with this command:

```
benchit benchmark scripts/hello_world.py --build-only
```

Which gives a result like:

```
Models discovered during profiling:

hello_world.py:
        pytorch_model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          SmallModel (<class 'hello_world.SmallModel'>)
                Location:       /home/jfowers/mlagility/examples/cli/scripts/hello_world.py, line 29
                Parameters:     55 (<0.1 MB)
                Hash:           479b1332
                Status:         Model successfully built!

pytorch_outputs: tensor([-0.1675,  0.1548, -0.1627,  0.0067,  0.3353], grad_fn=<AddBackward0>)

Woohoo! The 'benchmark' command is complete. Use the 'report' command to get a .csv file that summarizes results across all builds in the cache.
```

You can see that the model is discovered and built, but no benchmark took place.

> See the [Build Only documentation](https://github.com/groq/mlagility/blob/main/tools_user_guide.md#build-only) for more details.
