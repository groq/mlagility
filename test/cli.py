"""
Tests focused on the command-level functionality of benchit CLI
"""

import os
import glob
import csv
import argparse
from typing import List, Tuple, Any, Union
import unittest
from unittest.mock import patch
import sys
import io
from pathlib import Path
import shutil
from contextlib import redirect_stdout
import yaml
import onnx
from mlagility.cli.cli import main as benchitcli
import mlagility.cli.report as report
from mlagility.common import filesystem
import mlagility.api.ortmodel as ortmodel
import mlagility.api.trtmodel as trtmodel
import onnxflow.common.build as build
import onnxflow.common.cache as cache
import onnxflow.common.exceptions as exceptions


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

""",
    "linear2.py": """# labels: name::linear2 author::benchit license::mit test_group::b
import torch

torch.manual_seed(0)

# Define model class
class TwoLayerModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(TwoLayerModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)
        self.fc2 = torch.nn.Linear(output_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        output = self.fc2(output)
        return output


# Instantiate model and generate inputs
input_size = 10
output_size = 5
pytorch_model = TwoLayerModel(input_size, output_size)
inputs = {"x": torch.rand(input_size)}

pytorch_outputs = pytorch_model(**inputs)

# Print results
print(f"Pytorch_outputs: {pytorch_outputs}")
""",
    "crash.py": """# labels: name::crash author::benchit license::mit
import torch
import sys

torch.manual_seed(0)

# The purpose of this script is to intentionally crash
# so that we can test --resume
# Any test that doesn't supply the crash signal will treat this
# as a normal input script that runs a small model
if len(sys.argv) > 1:
    if sys.argv[1] == "crash!":
        assert False

class LinearTestModel(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(LinearTestModel, self).__init__()
        self.fc = torch.nn.Linear(input_features, output_features)

    def forward(self, x):
        output = self.fc(x)
        return output


input_features = 5
output_features = 5

# Model and input configurations
model = LinearTestModel(input_features, output_features)
inputs = {"x": torch.rand(input_features)}

output = model(**inputs)
""",
}


example_sequence_file = "example_sequence.py"

extras_dot_py = {
    example_sequence_file: """
from onnxflow.justbuildit.stage import Sequence
import onnxflow.justbuildit.export as export

def get_sequence():
    return Sequence(
        unique_name="example_sequence",
        monitor_message="Example pytorch sequence that only exports ONNX",
        stages=[
            export.ExportPytorchModel(),
            export.SuccessStage(),
        ],
        enable_model_validation=True,
    )
    """,
    "compiled.py": """
# labels: name::linear author::selftest test_group::selftest
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

# Compiled model
model = LinearTestModel(input_features, output_features)
model = torch.compile(model)
inputs = {"x": torch.rand(input_features)}
model(**inputs)

# Non-compiled model
model2 = LinearTestModel(input_features * 2, output_features)
inputs2 = {"x": torch.rand(input_features * 2)}
model2(**inputs2)
""",
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

for key, value in extras_dot_py.items():
    file_path = os.path.join(extras_dir, key)

    with open(file_path, "w", encoding="utf") as f:
        f.write(value)


def strip_dot_py(test_script_file: str) -> str:
    return test_script_file.split(".")[0]


def bash(cmd: str) -> List[str]:
    """
    Emulate behavior of bash terminal when listing files
    """
    return glob.glob(cmd)


def flatten(lst: List[Union[str, List[str]]]) -> List[str]:
    """
    Flatten List[Union[str, List[str]]] into a List[str]
    """
    flattened = []
    for element in lst:
        if isinstance(element, list):
            flattened.extend(element)
        else:
            flattened.append(element)
    return flattened


def assert_success_of_builds(
    test_script_files: List[str],
    cache_dir: str,
    info_property: Tuple[str, Any] = None,
    check_perf: bool = False,
    check_opset: int = None,
    runtime: str = "ort",
) -> int:
    # Figure out the build name by surveying the build cache
    # for a build that includes test_script_name in the name
    # TODO: simplify this code when
    # https://github.com/groq/mlagility/issues/16
    # is done
    builds = cache.get_all(cache_dir)
    builds_found = 0

    for test_script in test_script_files:
        test_script_name = strip_dot_py(test_script)
        script_build_found = False

        for build_state_file in builds:
            if test_script_name in build_state_file:
                build_state = build.load_state(state_path=build_state_file)
                assert build_state.build_status == build.Status.SUCCESSFUL_BUILD
                script_build_found = True
                builds_found += 1

                if info_property is not None:
                    assert (
                        build_state.info.__dict__[info_property[0]] == info_property[1]
                    ), f"{build_state.info.__dict__[info_property[0]]} == {info_property[1]}"

                if check_perf:
                    if runtime == "ort":
                        model_perf = ortmodel.ORTModel(
                            build_name=build_state.config.build_name,
                            cache_dir=build_state.cache_dir,
                        )
                    elif runtime == "trt":
                        model_perf = trtmodel.TRTModel(
                            build_name=build_state.config.build_name,
                            cache_dir=build_state.cache_dir,
                        )
                    else:
                        assert (
                            False
                        ), f"Runtime {runtime} has not been implemented as part of this function"
                    assert model_perf.mean_latency > 0
                    assert model_perf.throughput > 0

                if check_opset:
                    onnx_model = onnx.load(build_state.converted_onnx_file)
                    model_opset = getattr(onnx_model.opset_import[0], "version", None)
                    assert model_opset == check_opset

        assert script_build_found

    # Returns the total number of builds found
    return builds_found


class Testing(unittest.TestCase):
    def setUp(self) -> None:
        cache.rmdir(cache_dir)

        return super().setUp()

    def test_001_cli_single(self):
        # Test the first model in the corpus
        test_script = list(test_scripts_dot_py.keys())[0]

        testargs = [
            "benchit",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds([test_script], cache_dir)

    def test_002_search_multiple(self):
        # Test the first model in the corpus
        test_scripts = list(test_scripts_dot_py.keys())

        testargs = [
            "benchit",
            "benchmark",
            os.path.join(corpus_dir, test_scripts[0]),
            os.path.join(corpus_dir, test_scripts[1]),
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds([test_scripts[0], test_scripts[1]], cache_dir)

    def test_003_cli_build_dir(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_single

        test_scripts = test_scripts_dot_py.keys()

        testargs = [
            "benchit",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            benchitcli()

        assert_success_of_builds(test_scripts, cache_dir)

    def test_004_cli_report(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus

        test_scripts = test_scripts_dot_py.keys()

        # Build the test corpus so we have builds to report
        testargs = [
            "benchit",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            benchitcli()

        testargs = [
            "benchit",
            "cache",
            "report",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        # Read generated CSV file
        summary_csv_path = report.get_report_name()
        with open(summary_csv_path, "r", encoding="utf8") as summary_csv:
            summary = list(csv.DictReader(summary_csv))

        # Check if csv file contains all expected rows and columns
        expected_cols = [
            "model_name",
            "author",
            "model_class",
            "params",
            "hash",
            "license",
            "task",
            "groq_chips_used",
            "groq_estimated_latency",
            "nvidia_latency",
            "x86_latency",
            "onnx_exported",
            "onnx_optimized",
            "onnx_converted",
        ]
        linear_summary = summary[1]
        assert len(summary) == len(test_scripts)
        assert all(elem in expected_cols for elem in linear_summary)

        # Check whether all rows we expect to be populated are actually populated
        assert (
            linear_summary["model_name"] == "linear2"
        ), f"Wrong model name found {linear_summary['model_name']}"
        assert (
            linear_summary["author"] == "benchit"
        ), f"Wrong author name found {linear_summary['author']}"
        assert (
            linear_summary["model_class"] == "TwoLayerModel"
        ), f"Wrong class found {linear_summary['model_class']}"
        assert (
            linear_summary["hash"] == "80b93950"
        ), f"Wrong hash found {linear_summary['hash']}"
        assert (
            float(linear_summary["x86_latency"]) > 0
        ), f"x86 latency must be >0, got {linear_summary['x86_latency']}"
        assert (
            linear_summary["onnx_exported"] == "True"
        ), f"onnx_exported must be True, got {linear_summary['onnx_exported']}"
        assert (
            linear_summary["onnx_optimized"] == "True"
        ), f"onnx_optimized must be True, got {linear_summary['onnx_optimized']}"
        assert (
            linear_summary["onnx_converted"] == "True"
        ), f"onnx_converted must be True, got {linear_summary['onnx_converted']}"

    def test_005_cli_list(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus

        # Build the test corpus so we have builds to list
        testargs = [
            "benchit",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            benchitcli()

        # Make sure we can list the builds in the cache
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "benchit",
                "cache",
                "list",
                "--cache-dir",
                cache_dir,
            ]
            with patch.object(sys, "argv", testargs):
                benchitcli()

        for test_script in test_scripts_dot_py.keys():
            script_name = strip_dot_py(test_script)
            assert script_name in f.getvalue()

    def test_006_cli_delete(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus
        # - test_cli_list

        # Build the test corpus so we have builds to delete
        testargs = [
            "benchit",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            benchitcli()

        # Make sure we can list the builds in the cache
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "benchit",
                "cache",
                "list",
                "--cache-dir",
                cache_dir,
            ]
            with patch.object(sys, "argv", testargs):
                benchitcli()

        for test_script in test_scripts_dot_py.keys():
            script_name = strip_dot_py(test_script)
            assert script_name in f.getvalue()

        # Delete the builds
        testargs = [
            "benchit",
            "cache",
            "delete",
            "--all",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        # Make sure the builds are gone
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "benchit",
                "cache",
                "list",
                "--cache-dir",
                cache_dir,
            ]
            with patch.object(sys, "argv", testargs):
                benchitcli()

        for test_script in test_scripts_dot_py.keys():
            script_name = strip_dot_py(test_script)
            assert script_name not in f.getvalue()

    def test_007_cli_stats(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus

        # Build the test corpus so we have builds to print
        testargs = [
            "benchit",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            benchitcli()

        # Make sure we can print the builds in the cache
        for test_script in test_scripts_dot_py.keys():
            test_script_path = os.path.join(corpus_dir, test_script)
            builds, script_name = filesystem.get_builds_from_script(
                cache_dir, test_script_path
            )

            for build_name in builds:
                # Make sure each build can be accessed with `benchit cache stats`
                with redirect_stdout(io.StringIO()) as f:
                    testargs = [
                        "benchit",
                        "cache",
                        "stats",
                        build_name,
                        "--cache-dir",
                        cache_dir,
                    ]
                    with patch.object(sys, "argv", testargs):
                        benchitcli()

                    assert script_name in f.getvalue()

                # Make sure the MLAgility YAML file contains the fields
                # required for producing a report
                stats_file = os.path.join(
                    build.output_dir(cache_dir, build_name), "mlagility_stats.yaml"
                )
                with open(stats_file, "r", encoding="utf8") as stream:
                    stats_dict = yaml.load(stream, Loader=yaml.FullLoader)

                assert isinstance(stats_dict["hash"], str), stats_dict["hash"]
                assert isinstance(stats_dict["parameters"], int), stats_dict[
                    "parameters"
                ]
                assert isinstance(
                    stats_dict["onnx_input_dimensions"], dict
                ), stats_dict["onnx_input_dimensions"]
                assert isinstance(
                    stats_dict["onnx_model_information"], dict
                ), stats_dict["onnx_model_information"]
                assert isinstance(stats_dict["onnx_ops_counter"], dict), stats_dict[
                    "onnx_ops_counter"
                ]

                # Make sure the mlagility_stats has the expected ONNX opset
                assert (
                    stats_dict["onnx_model_information"]["opset"]
                    == build.DEFAULT_ONNX_OPSET
                ), stats_dict["onnx_model_information"]["opset"]

    def test_008_cli_version(self):
        # Get the version number
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "benchit",
                "version",
            ]
            with patch.object(sys, "argv", testargs):
                benchitcli()

        # Make sure we get back a 3-digit number
        assert len(f.getvalue().split(".")) == 3

    def test_009_cli_benchit_args(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_single

        # Test the first model in the corpus
        test_script = list(test_scripts_dot_py.keys())[0]

        # Set as many benchit args as possible
        testargs = [
            "benchit",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--rebuild",
            "always",
            "--groq-num-chips",
            "1",
            "--groqview",
            "--groq-compiler-flags=--large-program",
            "--groq-assembler-flags=--no-metrics",
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds([test_script], cache_dir)

    def test_010_cli_sequence(self):
        # Test the first model in the corpus
        test_script = list(test_scripts_dot_py.keys())[0]

        testargs = [
            "benchit",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--build-only",
            "--sequence-file",
            os.path.join(extras_dir, example_sequence_file),
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds(
            [test_script],
            cache_dir,
            ("all_build_stages", ["export_pytorch", "set_success"]),
        )

    def test_011_cli_benchmark(self):
        # Test the first model in the corpus
        test_script = list(test_scripts_dot_py.keys())[0]

        testargs = [
            "benchit",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds([test_script], cache_dir, None, check_perf=True)

    def test_012_cli_resume(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_single
        # - test_cli_build_dir

        test_scripts = test_scripts_dot_py.keys()

        # Build all the scripts, sending a crash signal to crash.py
        with self.assertRaises(AssertionError):
            testargs = [
                "benchit",
                "benchmark",
                bash(f"{corpus_dir}/*.py"),
                "--build-only",
                "--cache-dir",
                cache_dir,
                "--script-args",
                "crash!",
            ]
            with patch.object(sys, "argv", flatten(testargs)):
                benchitcli()

        testargs = [
            "benchit",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--build-only",
            "--cache-dir",
            cache_dir,
            "--script-args",
            "crash!",
            "--resume",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            benchitcli()

        # All builds except for crash.py should have succeeded
        test_scripts = [x for x in test_scripts if x != "crash.py"]
        assert_success_of_builds(test_scripts, cache_dir)

    # TODO: Investigate why this test is non-deterministically failing
    @unittest.skip("Flaky test")
    def test_013_cli_labels(self):
        # Only build models labels with test_group::a
        testargs = [
            "benchit",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--labels",
            "test_group::a",
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            benchitcli()

        state_files = [Path(p).stem for p in cache.get_all(cache_dir)]
        assert state_files == ["linear_d5b1df11_state"]

        # Delete the builds
        testargs = [
            "benchit",
            "cache",
            "delete",
            "--all",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert cache.get_all(cache_dir) == []

        # Only build models labels with test_group::a and test_group::b
        testargs = [
            "benchit",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--labels",
            "test_group::a,b",
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            benchitcli()

        state_files = [Path(p).stem for p in cache.get_all(cache_dir)]
        assert state_files == ["linear_d5b1df11_state", "linear2_80b93950_state"]

    # TODO: Re-implement to be more robust. Current implementation relies on
    # coincidence (whether or not Groq HW is installed) to function.
    @unittest.skip("Needs re-implementation")
    def test_014_report_on_failed_build(self):
        # Run benchit on groq device (will fail since the HW is not available)
        testargs = [
            "benchit",
            bash(f"{corpus_dir}/linear.py"),
            "--device",
            "groq",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            benchitcli()

        # Ensure test failed
        build_state = build.load_state(state_path=cache.get_all(cache_dir)[0])
        assert build_state.build_status != build.Status.SUCCESSFUL_BUILD

        # Generate report
        testargs = [
            "benchit",
            "cache",
            "report",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        # Read generated CSV file
        summary_csv_path = report.get_report_name()
        summary = None
        with open(summary_csv_path, "r", encoding="utf8") as summary_csv:
            summary = list(csv.DictReader(summary_csv))

        # Ensure parameters and hash have been saved despite crash
        assert (
            len(summary) == 1
        ), "Report must contain only one row, but got {len(summary)}"
        assert (
            summary[0]["params"] == "110"
        ), "Wrong number of parameters found in report"
        assert summary[0]["hash"] == "d5b1df11", "Wrong hash found in report"

    def test_015_runtimes(self):
        # Attempt to benchmark using an invalid runtime
        with self.assertRaises(argparse.ArgumentError):
            testargs = [
                "benchit",
                "benchmark",
                bash(f"{corpus_dir}/linear.py"),
                "--cache-dir",
                cache_dir,
                "--device",
                "x86",
                "--runtime",
                "trt",
            ]
            with patch.object(sys, "argv", flatten(testargs)):
                benchitcli()

        # Benchmark with Pytorch
        testargs = [
            "benchit",
            "benchmark",
            bash(f"{corpus_dir}/linear.py"),
            "--cache-dir",
            cache_dir,
            "--device",
            "x86",
            "--runtime",
            "torch-eager",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            benchitcli()

        # Benchmark with Onnx Runtime
        testargs = [
            "benchit",
            "benchmark",
            bash(f"{corpus_dir}/linear.py"),
            "--cache-dir",
            cache_dir,
            "--device",
            "x86",
            "--runtime",
            "ort",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            benchitcli()

        # Ensure that we have two benchmarking results as part of the state file
        stats_file = os.path.join(
            build.output_dir(cache_dir, "linear_benchit_d5b1df11"),
            "mlagility_stats.yaml",
        )
        with open(stats_file, "r", encoding="utf8") as stream:
            stats_dict = yaml.load(stream, Loader=yaml.FullLoader)
        assert len(stats_dict["performance"]) == 2

    def test_016_cli_onnx_opset(self):
        # Test the first model in the corpus
        test_script = list(test_scripts_dot_py.keys())[0]

        user_opset = 15
        assert user_opset != build.DEFAULT_ONNX_OPSET

        testargs = [
            "benchit",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
            "--onnx-opset",
            str(user_opset),
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds(
            [test_script], cache_dir, None, check_perf=True, check_opset=user_opset
        )

    def test_017_cli_process_isolation(self):
        # Test the first model in the corpus
        test_script = list(test_scripts_dot_py.keys())[0]

        testargs = [
            "benchit",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
            "--process-isolation",
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds([test_script], cache_dir, None, check_perf=True)

    def test_018_skip_compiled(self):
        test_script = "compiled.py"
        testargs = [
            "benchit",
            "benchmark",
            os.path.join(extras_dir, test_script),
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        builds_found = assert_success_of_builds([test_script], cache_dir)

        # Compile.py contains two Pytorch models.
        # One of those is compiled and should be skipped.
        assert builds_found == 1

    def test_019_invalid_file_type(self):
        # Ensure that we get an error when running benchit with invalid input_files
        with self.assertRaises(exceptions.ArgError):
            testargs = ["benchit", "gobbledegook"]
            with patch.object(sys, "argv", flatten(testargs)):
                benchitcli()


if __name__ == "__main__":
    unittest.main()
