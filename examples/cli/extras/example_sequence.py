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