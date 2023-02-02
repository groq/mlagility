import os
import unittest
from pathlib import Path
import shutil
import subprocess
import numpy as np
import mlagility.common.filesystem as filesystem
import groqflow.common.cache as cache

# We generate a corpus on to the filesystem during the test
# to get around how weird bake tests are when it comes to
# filesystem access

test_models_dot_py = {
    "linear_pytorch": """
import torch
import argparse

torch.manual_seed(0)

# Receive command line arg
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--my-arg",
)
args = parser.parse_args()
print(f"Received arg {args.my_arg}")

class LinearTestModel(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(LinearTestModel, self).__init__()
        self.fc = torch.nn.Linear(input_features, output_features)

    def forward(self, x):
        output = self.fc(x)
        return output

input_features = 10
output_features = 10

# Model and input configurations
model = LinearTestModel(input_features, output_features)
unexecuted_model = LinearTestModel(input_features+1, output_features)
inputs = {"x": torch.rand(input_features)}
output = model(**inputs)

""",
    "linear_keras": """
import tensorflow as tf

tf.random.set_seed(0)

# Define model class
class SmallKerasModel(tf.keras.Model):  # pylint: disable=abstract-method
    def __init__(self, output_size):
        super(SmallKerasModel, self).__init__()
        self.dense = tf.keras.layers.Dense(output_size, activation="relu")

    def call(self, x):  # pylint: disable=arguments-differ
        output = self.dense(x)
        return output


# Instantiate model and generate inputs
batch_size = 1
input_size = 10
output_size = 5
keras_model = SmallKerasModel(output_size)

inputs = {"x": tf.random.uniform((batch_size, input_size), dtype=tf.float32)}

keras_outputs = keras_model(**inputs)

""",
    "pipeline": """
from transformers import (
    TextClassificationPipeline,
    BertForSequenceClassification,
    BertConfig,
    PreTrainedTokenizerFast,
)


class MyPipeline(TextClassificationPipeline):
    def __init__(self, **kwargs):
        configuration = BertConfig()
        tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
        super().__init__(
            model=BertForSequenceClassification(configuration), tokenizer=tokenizer
        )


my_pipeline = MyPipeline()
my_pipeline("This restaurant is awesome")
""",
    "activation": """
import torch
m = torch.nn.GELU()
input = torch.randn(2)
output = m(input)
""",
    "encoder_decoder": """
import transformers
import torch

torch.manual_seed(0)

config = transformers.SpeechEncoderDecoderConfig.from_encoder_decoder_configs(
    transformers.Wav2Vec2Config(), transformers.BertConfig()
)
model = transformers.SpeechEncoderDecoderModel(config)
inputs = {
    "decoder_input_ids": torch.ones(1, 64, dtype=torch.long),
    "input_values": torch.ones(1, 10000, dtype=torch.float),
}

model(**inputs)""",
}
minimal_tokenizer = """
{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0
    }
  }
}"""

# Create a test directory
test_dir = "autogroq_test_dir"
dirpath = Path(test_dir)
if dirpath.is_dir():
    shutil.rmtree(dirpath)
os.makedirs(test_dir)

# Add files to test directory
for key, value in test_models_dot_py.items():
    model_path = os.path.join(test_dir, f"{key}.py")
    with open(model_path, "w", encoding="utf") as f:
        f.write(value)
with open(os.path.join(test_dir, "tokenizer.json"), "w", encoding="utf") as f:
    f.write(minimal_tokenizer)


# Change CWD
os.chdir(test_dir)


def run_analysis(args):

    # Running cli
    output = subprocess.check_output(args, encoding="UTF-8")

    # Process outputs
    output = output[output.rfind("Models found") :]
    models_executed = output.count("(executed")
    models_built = output.count("Model successfully built!")
    return models_executed, 0, models_built


class Testing(unittest.TestCase):
    def setUp(self) -> None:
        cache.rmdir(filesystem.DEFAULT_CACHE_DIR)
        return super().setUp()

    def test_01_basic(self):
        pytorch_output = run_analysis(
            ["autogroq", "linear_pytorch.py", "--analyze-only"]
        )
        assert np.array_equal(pytorch_output, (1, 0, 0))

    def test_02_basic_keras(self):
        keras_output = run_analysis(["autogroq", "linear_keras.py", "--analyze-only"])
        assert np.array_equal(keras_output, (1, 0, 0))

    def test_03_depth(self):
        # Depth is only tested for Pytorch, since Keras has no max_depth support
        output = run_analysis(
            ["autogroq", "linear_pytorch.py", "-d1", "--analyze-only"]
        )
        assert np.array_equal(output, (2, 0, 0))

    def test_04_build(self):
        output = run_analysis(
            ["autogroq", "linear_pytorch.py", "-d1", "--targets", "60931adb"]
        )
        assert np.array_equal(output, (2, 0, 1))

    def test_05_build_keras(self):
        output = run_analysis(["autogroq", "linear_keras.py"])
        assert np.array_equal(output, (1, 0, 1))

    def test_06_cache(self):
        model_hash = "60931adb"
        cache_dir = "cache-dir"
        run_analysis(
            [
                "autogroq",
                "linear_pytorch.py",
                "-d1",
                "--targets",
                model_hash,
                "--cache-dir",
                cache_dir,
                "--lean-cache",
            ]
        )
        files = os.listdir(f"{cache_dir}/linear_pytorch_{model_hash}")
        cache_is_lean = len([x for x in files if ".onnx" in x]) == 0
        metadata_found = len([x for x in files if ".txt" in x]) > 0
        assert metadata_found and cache_is_lean

    def test_07_args(self):
        output = subprocess.check_output(
            [
                "autogroq",
                "linear_pytorch.py",
                "-d1",
                "--input-args",
                "--my-arg test_arg",
                "--analyze-only",
            ],
            encoding="UTF-8",
        )
        assert "Received arg test_arg" in output

    def test_08_pipeline(self):
        output = run_analysis(["autogroq", "pipeline.py", "--analyze-only"])
        assert np.array_equal(output, (1, 0, 0))

    def test_09_activation(self):
        output = run_analysis(["autogroq", "activation.py", "--analyze-only"])
        assert np.array_equal(output, (0, 0, 0))

    def test_10_encoder_decoder(self):
        output = run_analysis(["autogroq", "encoder_decoder.py", "--analyze-only"])
        assert np.array_equal(output, (1, 0, 0))

    def test_11_benchit_hashes(self):
        output = run_analysis(
            [
                "benchit",
                "linear_pytorch.py:60931adb",
                "--build-only",
                "--max-depth",
                "1",
            ]
        )
        assert np.array_equal(output, (2, 0, 1))


if __name__ == "__main__":
    unittest.main()
