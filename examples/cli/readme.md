# Learning the `benchit` CLI

This document is a tutorial for exploring the different features of the `benchit` command line interface (CLI). You can learn the details of those features in the [Tools User Guide](https://github.com/groq/mlagility/blob/main/docs/tools_user_guide.md) and learn about their implementation in the [Code Organization Guide](https://github.com/groq/mlagility/blob/main/docs/code.md).

We've created this tutorial document because `benchit` is a CLI that benchmarks the contents of `.py` scripts. So all of the `.py` scripts in the `examples/cli/scripts` directory are meant to be fed into `benchit` to demonstrate some specific functionality.

Once you've familiarized yourself with these features, head over to the [`models` directory](https://github.com/groq/mlagility/tree/main/models) to learn how to use `benchit` with real world machine learning models.

## Table of Contents

TODO

## Tutorials

All of the tutorials assume that your command line is in the same location as this readme file (`examples/cli`).

### Hello World

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

### Keras

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

### Nvidia Benchmarking

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

### Benchmarking Two Models from One Script

The MLAgility tools will benchmark all models discovered in the input script. We can demonstrate this with the `two_models.py` script.

Run the following command:

```
benchit scripts/two_models.py
```

To get a result like:

```