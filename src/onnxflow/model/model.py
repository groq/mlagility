import os
from typing import Dict, List, Any
import numpy as np
import torch

import onnxflow.common.exceptions as exp
import onnxflow.common.build as build
import onnxflow.common.tensor_helpers as tensor_helpers
import onnxflow.common.tf_helpers as tf_helpers


class BaseModel:
    def __init__(self, state: build.State, tensor_type=np.array, input_dtypes=None):

        self.input_dtypes = input_dtypes
        self.tensor_type = tensor_type
        self.state = state
        self.remote_client = None
        self.log_execute_path = os.path.join(
            build.output_dir(state.cache_dir, self.state.config.build_name),
            "log_execute.txt",
        )

        # checked_dependencies is a persistent state across executions
        # so that we only need to check the dependencies once
        self.checked_dependencies = False

    def _validate_input_collection(self, input_collection, function_name) -> None:
        if input_collection is None:
            raise exp.ModelArgError(
                (
                    f"Model.{function_name}() received an input_collection with type "
                    f"{type(input_collection)}, however the input_collection arg must be "
                    "a collection of dictionaries."
                )
            )
        else:
            if len(input_collection) == 0:
                raise exp.ModelArgError(
                    f"Model.{function_name}() received an empty collection as input."
                )

        # Check whether all elements of input_collection have the shape required by the model
        for inputs in input_collection:
            self._validate_inputs(inputs, function_name, True)

    def _validate_inputs(self, inputs, function_name, from_collection=False) -> None:
        if from_collection:
            collection_msg = "input_collection "
        else:
            collection_msg = ""

        if not isinstance(inputs, dict):
            raise exp.ModelArgError(
                (
                    f"Model.{function_name}() {collection_msg}received inputs of type "
                    f"{type(inputs)}, however the inputs must be a dictionary."
                )
            )

        # Check whether the inputs provided have the shapes required by the model's
        # forward function
        tensor_helpers.check_shapes_and_dtypes(
            inputs,
            self.state.expected_input_shapes,
            self.state.expected_input_dtypes,
            self.state.downcast_applied
        )

    # Models with a single output are returned as either a torch.tensor,
    # tf.Tensor, or an np.array (see tensor_type)
    # Models with multiple outputs are returned as either a tuple of
    # torch.tensors, tf.Tensors, or np.arrays
    def _unpack_results(self, results: List[Dict], output_nodes, num_outputs):
        if tf_helpers.type_is_tf_tensor(self.tensor_type):
            # pylint: disable=import-error
            import tensorflow

            unpacked_results = [
                tensorflow.convert_to_tensor(results[x]) for x in output_nodes
            ]
        else:
            unpacked_results = [self.tensor_type(results[x]) for x in output_nodes]
        return unpacked_results[0] if num_outputs == 1 else tuple(unpacked_results)

    def _unpack_results_file(self, packed_results: str) -> Any:
        """
        Unpack execution results from a file
        """

        np_result = np.load(packed_results, allow_pickle=True)

        # Ensure that the output nodes generated are the same as the expected output nodes
        output_nodes = self.state.expected_output_names
        num_outputs = len(output_nodes)
        output_nodes_received = list(np_result[0].keys())
        if not all(node in output_nodes for node in output_nodes_received):
            raise exp.ModelRuntimeError(
                (
                    f"Model expected outputs {str(self.state.expected_output_names)} "
                    f"but got {str(output_nodes_received)}"
                )
            )

        # Unpack all results from the collection and pack them in a list
        unpacked_result_list = [
            self._unpack_results(output_sample, output_nodes, num_outputs)
            for output_sample in np_result
        ]

        # If a collection of inputs was received, return a list of results
        # If a single set of inputs was received, return a single result
        if len(np_result) > 1:
            return unpacked_result_list
        else:
            return unpacked_result_list[0]


class PytorchModelWrapper(BaseModel):
    def __init__(self, state):
        tensor_type = torch.tensor
        super(PytorchModelWrapper, self).__init__(state, tensor_type)


class KerasModelWrapper(BaseModel):
    def __init__(self, state):
        # pylint: disable=import-error
        import tensorflow

        tensor_type = tensorflow.Tensor
        super(KerasModelWrapper, self).__init__(state, tensor_type)


class HummingbirdWrapper(BaseModel):
    def __init__(self, state):
        super(HummingbirdWrapper, self).__init__(
            state, input_dtypes={"input_0": "float32"}
        )

    def _unpack_results(self, results: List[Dict], output_nodes, num_outputs):
        unpacked_results = [
            self.tensor_type(results[k]) for k in output_nodes if k != "variable"
        ]
        unpacked_results.insert(0, self.tensor_type(results["variable"]))
        return unpacked_results[0] if num_outputs == 1 else tuple(unpacked_results)


def load(build_name: str, cache_dir=build.DEFAULT_CACHE_DIR) -> BaseModel:
    state = build.load_state(cache_dir=cache_dir, build_name=build_name)

    if state.model_type == build.ModelType.PYTORCH:
        return PytorchModelWrapper(state)
    elif state.model_type == build.ModelType.KERAS:
        return KerasModelWrapper(state)
    elif state.model_type == build.ModelType.HUMMINGBIRD:
        return HummingbirdWrapper(state)
    else:
        return BaseModel(state)
