"""
This script is an example of a sequence.py file. Such a sequence.py file
can be used to redefine the build phase of the benchit CLI, benchmark_script(),
and benchmark_model() to have any custom behavior.

In this example sequence.py file we are setting the build sequence to simply
export from pytorch to ONNX. This differs from the default build sequence, which
exports to ONNX, optimizes, and converts to float16.

You can pass this file into benchit with a command like:

    benchit benchmark INPUT_SCRIPTS --sequence-file example_sequence.py --build-only
"""

from groqflow.justgroqit.stage import Sequence
import groqflow.justgroqit.export as export
from mlagility.common.groqflow_helpers import SuccessStage


def get_sequence():
    return Sequence(
        unique_name="example_sequence",
        monitor_message="Example pytorch sequence that only exports ONNX",
        stages=[
            export.ExportPytorchModel(),
            SuccessStage(),
        ],
        enable_model_validation=True,
    )
