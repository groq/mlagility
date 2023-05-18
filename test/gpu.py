"""
MLAgility GPU tests
"""

import os
import unittest
from unittest.mock import patch
import sys
from pathlib import Path
import shutil
import yaml
from mlagility.cli.cli import main as benchitcli
import onnxflow.common.build as build
import onnxflow.common.cache as cache
from cli import assert_success_of_builds, flatten, bash, strip_dot_py

# We generate a corpus on to the filesystem during the test
# to get around how weird bake tests are when it comes to
# filesystem access

test_scripts_dot_py = {
    "linear.py": """# labels: name::linear author::benchit license::mit test_group::a
import torch

torch.manual_seed(0)


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
inputs = {"x": torch.rand(input_features)}

output = model(**inputs)

"""
}


# Create a test directory and make it the CWD
test_dir = "cli_test_dir"
cache_dir_name = "cache-dir"
cache_dir = os.path.abspath(cache_dir_name)
dirpath = Path(test_dir)
if dirpath.is_dir():
    shutil.rmtree(dirpath)
os.makedirs(test_dir)
os.chdir(test_dir)

corpus_dir = os.path.join(os.getcwd(), "test_corpus")
extras_dir = os.path.join(corpus_dir, "extras")
os.makedirs(extras_dir, exist_ok=True)

for key, value in test_scripts_dot_py.items():
    model_path = os.path.join(corpus_dir, key)

    with open(model_path, "w", encoding="utf") as f:
        f.write(value)


class Testing(unittest.TestCase):
    def setUp(self) -> None:
        cache.rmdir(cache_dir)

        return super().setUp()

    def test_basic(self):
        test_script = "linear.py"
        # Benchmark with Pytorch
        testargs = [
            "benchit",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
            "--device",
            "nvidia",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            benchitcli()

        assert_success_of_builds(
            [test_script], cache_dir, check_perf=True, runtime="trt"
        )


if __name__ == "__main__":
    unittest.main()
