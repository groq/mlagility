# Customizing Builds

This chapter of the `benchit` CLI tutorial focuses on techniques to customize the behavior of your `benchit` builds. You will learn things such as:
- [How to build models without benchmarking them](#build-only)
- [How to customize the build process with Sequences](#sequence-file)

The tutorial chapters are:
1. [Getting Started](https://github.com/groq/mlagility/blob/main/examples/cli/readme.md)
1. [Guiding Model Discovery](https://github.com/groq/mlagility/blob/main/examples/cli/discovery.md): `benchit` arguments that customize the model discovery process to help streamline your workflow.
1. [Working with the Cache](https://github.com/groq/mlagility/blob/main/examples/cli/cache.md): `benchit` arguments and commands that help you understand, inspect, and manipulate the `mlagility cache`.
1. Customizing Builds (this document): `benchit` arguments that customize build behavior to unlock new workflows.

# Build Tutorials

All of the tutorials assume that your current working directory is in the same location as this readme file (`examples/cli`).

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

Woohoo! The 'benchmark' command is complete.
```

You can see that the model is discovered and built, but no benchmark took place.

> See the [Build Only documentation](https://github.com/groq/mlagility/blob/main/tools_user_guide.md#build-only) for more details.

## Sequence File

You can customize the behavior of the [Build](https://github.com/groq/mlagility/blob/main/tools_user_guide.md#build) stage of `benchit` by creating a custom `Sequence`.

A `Sequence` tells the `benchmark_model()` API within `benchit` how to `build` a model to prepare it for benchmarking.

The default `Sequence` for CPU and GPU benchmarking performs the following build steps:
1. Export the model to an ONNX file
1. Use ONNX Runtime to optimize the ONNX file
1. Use ONNX ML Tools to the convert the optimized ONNX file to float16
1. Set the `build_status=successful_build` property

You can see this if you already did the [Hello World tutorial](https://github.com/groq/mlagility/blob/main/examples/cli/readme.md#hello-world) by running the command:

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

# Thanks!

Now that you have completed this tutorial, make sure to check out the other tutorials if you want to learn more:
1. [Getting Started](https://github.com/groq/mlagility/blob/main/examples/cli/readme.md)
1. [Guiding Model Discovery](https://github.com/groq/mlagility/blob/main/examples/cli/discovery.md): `benchit` arguments that customize the model discovery process to help streamline your workflow.
1. [Working with the Cache](https://github.com/groq/mlagility/blob/main/examples/cli/cache.md): `benchit` arguments and commands that help you understand, inspect, and manipulate the `mlagility cache`.
1. Customizing Builds (this document): `benchit` arguments that customize build behavior to unlock new workflows.