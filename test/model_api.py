import os
import unittest
import torch
import shutil
from pathlib import Path
import tensorflow as tf
import groqflow.justgroqit.stage as stage
import groqflow.common.cache as cache
import groqflow.justgroqit.export as export
import groqflow.common.build as build
from mlagility import benchmark_model


class SmallPytorchModel(torch.nn.Module):
    def __init__(self):
        super(SmallPytorchModel, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

    def forward(self, x):
        output = self.fc(x)
        return output


class AnotherSimplePytorchModel(torch.nn.Module):
    def __init__(self):
        super(AnotherSimplePytorchModel, self).__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.relu(x)
        return output


class SmallKerasModel(tf.keras.Model):  # pylint: disable=abstract-method
    def __init__(self):
        super(SmallKerasModel, self).__init__()
        self.dense = tf.keras.layers.Dense(10)

    def call(self, x):  # pylint: disable=arguments-differ
        return self.dense(x)


# Define pytorch model and inputs
pytorch_model = SmallPytorchModel()
tiny_pytorch_model = AnotherSimplePytorchModel()
inputs = {"x": torch.rand(10)}
inputs_2 = {"x": torch.rand(5)}
input_tensor = torch.rand(10)

# Define keras models and inputs
batch_keras_inputs = {"x": tf.random.uniform((1, 10), dtype=tf.float32)}
keras_subclass_model = SmallKerasModel()
keras_subclass_model.build(input_shape=(1, 10))


# Create a test directory
test_dir = "model_api_test_dir"
cache_dir = "cache-dir"
dirpath = Path(test_dir)
if dirpath.is_dir():
    shutil.rmtree(dirpath)
os.makedirs(test_dir)

# Change CWD
os.chdir(test_dir)


def get_build_state(cache_dir, build_name):
    return build.load_state(cache_dir=cache_dir, build_name=build_name)


class Testing(unittest.TestCase):
    def setUp(self) -> None:
        cache.rmdir(cache_dir)
        return super().setUp()

    def test_001_build_pytorch_model(self):
        build_name = "build_pytorch_model"
        benchmark_model(
            pytorch_model,
            inputs,
            build_name=build_name,
            rebuild="always",
            build_only=True,
            cache_dir=cache_dir,
        )
        state = get_build_state(cache_dir, build_name)
        assert state.build_status == build.Status.SUCCESSFUL_BUILD

    def test_002_build_keras_model(self):
        build_name = "full_compilation_keras_subclass_model"
        benchmark_model(
            keras_subclass_model,
            batch_keras_inputs,
            build_name=build_name,
            rebuild="always",
            build_only=True,
            cache_dir=cache_dir,
        )
        state = get_build_state(cache_dir, build_name)
        assert state.build_status == build.Status.SUCCESSFUL_BUILD

    def test_003_custom_stage(self):
        build_name = "custom_stage"

        class MyCustomStage(stage.GroqitStage):
            def __init__(self, funny_saying):
                super().__init__(
                    unique_name="funny_stage",
                    monitor_message="Funny Stage",
                )

                self.funny_saying = funny_saying

            def fire(self, state):

                print(f"funny message: {self.funny_saying}")
                state.build_status = build.Status.SUCCESSFUL_BUILD
                return state

        my_custom_stage = MyCustomStage(
            funny_saying="Is a fail whale a fail at all if it makes you smile?"
        )
        my_sequence = stage.Sequence(
            unique_name="my_sequence",
            monitor_message="Running My Sequence",
            stages=[
                export.ExportPytorchModel(),
                export.OptimizeOnnxModel(),
                my_custom_stage,
            ],
        )

        benchmark_model(
            pytorch_model,
            inputs,
            build_name=build_name,
            rebuild="always",
            sequence=my_sequence,
            build_only=True,
            cache_dir=cache_dir,
        )

        state = get_build_state(cache_dir, build_name)
        return state.build_status == build.Status.SUCCESSFUL_BUILD

    def test_004_local_benchmark(self):
        build_name = "local_benchmark"
        perf = benchmark_model(
            pytorch_model,
            inputs,
            device="x86",
            backend="local",
            build_name=build_name,
            rebuild="always",
            cache_dir=cache_dir,
        )
        state = get_build_state(cache_dir, build_name)
        assert state.build_status == build.Status.SUCCESSFUL_BUILD

        assert perf.mean_latency > 0
        assert perf.throughput > 0


if __name__ == "__main__":
    unittest.main()
