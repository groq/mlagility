from typing import Optional, List, Tuple, Union, Dict, Any, Type, Callable
from collections.abc import Collection
import sys
import os
import torch
import onnx
import onnxflow.common.build as build
import onnxflow.common.cache as cache
import onnxflow.common.exceptions as exp
import onnxflow.common.printing as printing
import onnxflow.common.tf_helpers as tf_helpers
import onnxflow.common.onnx_helpers as onnx_helpers
import onnxflow.common.tensor_helpers as tensor_helpers
import onnxflow.justbuildit.export as export
import onnxflow.justbuildit.stage as stage
import onnxflow.justbuildit.hummingbird as hummingbird
from onnxflow.version import __version__ as onnxflow_version

polish_onnx_sequence = stage.Sequence(
    "polish_onnx_sequence",
    "Polishing ONNX file",
    [
        export.OptimizeOnnxModel(),
        export.ConvertOnnxToFp16(),
        export.SuccessStage(),
    ],
)

default_pytorch_export_sequence = stage.Sequence(
    "default_pytorch_export_sequence",
    "Exporting PyTorch Model",
    [
        export.ExportPytorchModel(),
        export.SuccessStage(),
    ],
)

default_pytorch_sequence = stage.Sequence(
    "default_pytorch_export_sequence",
    "Exporting PyTorch Model",
    [export.ExportPytorchModel(), polish_onnx_sequence],
)

pytorch_sequence_with_quantization = stage.Sequence(
    "pytorch_export_sequence_with_quantization",
    "Exporting PyTorch Model and Quantizing Exported ONNX",
    [
        export.ExportPytorchModel(),
        export.OptimizeOnnxModel(),
        export.QuantizeONNXModel(),
        export.SuccessStage(),
    ],
)

default_keras_export_sequence = stage.Sequence(
    "default_keras_export_sequence",
    "Building Keras Model",
    [
        export.ExportKerasModel(),
        export.SuccessStage(),
    ],
)

default_keras_sequence = stage.Sequence(
    "default_keras_sequence",
    "Building Keras Model",
    [
        export.ExportKerasModel(),
        polish_onnx_sequence,
    ],
)

default_onnx_export_sequence = stage.Sequence(
    "default_onnx_export_sequence",
    "Building ONNX Model",
    [
        export.ReceiveOnnxModel(),
        export.SuccessStage(),
    ],
)


default_onnx_sequence = stage.Sequence(
    "default_onnx_sequence",
    "Building ONNX Model",
    [
        export.ReceiveOnnxModel(),
        polish_onnx_sequence,
    ],
)

default_hummingbird_export_sequence = stage.Sequence(
    "default_hummingbird_export_sequence",
    "Building Hummingbird Model",
    [
        hummingbird.ConvertHummingbirdModel(),
        export.SuccessStage(),
    ],
)

default_hummingbird_sequence = stage.Sequence(
    "default_hummingbird_sequence",
    "Building Hummingbird Model",
    [
        hummingbird.ConvertHummingbirdModel(),
        export.OptimizeOnnxModel(),
        export.SuccessStage(),
    ],
)

default_compiler_flags = []

default_assembler_flags = [
    "--ifetch-from-self",
    "--ifetch-slice-ordering=round-robin",
]


def lock_config(
    model: build.UnionValidModelInstanceTypes,
    build_name: Optional[str] = None,
    sequence: stage.Sequence = None,
    onnx_opset: Optional[int] = None,
) -> build.Config:

    """
    Process the user's configuration arguments to build_model():
    1. Raise exceptions for illegal arguments
    2. Replace unset arguments with default values
    3. Lock the configuration into an immutable object
    """

    # The default model name is the name of the python file that calls build_model()
    auto_name = False
    if build_name is None:
        build_name = sys.argv[0].split("/")[-1].split(".")[0]
        auto_name = True

    if sequence is None:
        # The value ["default"] indicates that build_model() will be assigning some
        # default sequence later in the program
        stage_names = ["default"]
    else:
        stage_names = sequence.get_names()

    # Detect and validate ONNX opset
    if isinstance(model, str) and model.endswith(".onnx"):
        onnx_file_opset = onnx_helpers.get_opset(onnx.load(model))

        if onnx_opset is not None and onnx_opset != onnx_file_opset:
            raise ValueError(
                "When using a '.onnx' file as input, the onnx_opset argument must "
                "be None or exactly match the ONNX opset of the '.onnx' file. However, the "
                f"'.onnx' file has opset {onnx_file_opset}, while onnx_opset was set "
                f"to {onnx_opset}"
            )

        opset_to_use = onnx_file_opset
    else:
        if onnx_opset is None:
            opset_to_use = build.DEFAULT_ONNX_OPSET
        else:
            opset_to_use = onnx_opset

    # Store the args that should be immutable
    config = build.Config(
        build_name=build_name,
        auto_name=auto_name,
        sequence=stage_names,
        onnx_opset=opset_to_use,
    )

    return config


def decode_version_number(version: str) -> Dict[str, int]:
    numbers = [int(x) for x in version.split(".")]
    return {"major": numbers[0], "minor": numbers[1], "patch": numbers[0]}


def validate_cached_model(
    config: build.Config,
    model_type: build.ModelType,
    state: build.State,
    model: build.UnionValidModelInstanceTypes = None,
    inputs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Verify whether anything in the call to build_model() changed
    We require the user to resolve the discrepancy when such a
    change occurs, so the purpose of this function is simply to
    detect these conditions and raise an appropriate error.
    If this function returns without raising an exception then
    the cached model is valid to use in the build.
    """

    result = []

    current_version_decoded = decode_version_number(onnxflow_version)
    state_version_decoded = decode_version_number(state.onnxflow_version)

    out_of_date: Union[str, bool] = False
    if current_version_decoded["major"] > state_version_decoded["major"]:
        out_of_date = "major"
    elif current_version_decoded["minor"] > state_version_decoded["minor"]:
        out_of_date = "minor"

    if out_of_date:
        msg = (
            f"Your build {state.config.build_name} was previously built against "
            f"onnxflow version {state.onnxflow_version}, "
            f"however you are now using onxxflow version {onnxflow_version}. The previous build is "
            f"incompatible with this version of onnxflow, as indicated by the {out_of_date} "
            "version number changing. See **docs/versioning.md** for details."
        )
        result.append(msg)

    if model is not None:
        model_changed = state.model_hash != build.hash_model(model, model_type)
    else:
        model_changed = False

    if inputs is not None:
        input_shapes_changed, input_dtypes_changed = tensor_helpers.check_shapes_and_dtypes(
                inputs,
                state.expected_input_shapes,
                state.expected_input_dtypes,
                expect_downcast=state.downcast_applied,
                raise_error=False
            )
    else:
        input_shapes_changed = False
        input_dtypes_changed = False

    changed_args = []
    for key in vars(state.config):
        if vars(config)[key] != vars(state.config)[key]:
            changed_args.append((key, vars(config)[key], vars(state.config)[key]))

    # Show an error if the model changed
    build_conditions_changed = (
        model_changed
        or input_shapes_changed
        or input_dtypes_changed
        or len(changed_args) > 0
    )
    if build_conditions_changed:
        # Show an error if build_name is not specified for different models on the same script
        if (
            state.uid == build.unique_id()
            and state.build_status != build.Status.PARTIAL_BUILD
        ):
            msg = (
                "You are building multiple different models in the same script "
                "without specifying a unique build_model(..., build_name=) for each build."
            )
            result.append(msg)

        if model_changed:
            msg = (
                f'Model "{config.build_name}" changed since the last time it was built.'
            )
            result.append(msg)

        if input_shapes_changed:
            input_shapes, _ = build.get_shapes_and_dtypes(inputs)
            msg = (
                f'Input shape of model "{config.build_name}" changed from '
                f"{state.expected_input_shapes} to {input_shapes} "
                f"since the last time it was built."
            )
            result.append(msg)

        if input_dtypes_changed:
            _, input_dtypes = build.get_shapes_and_dtypes(inputs)
            msg = (
                f'Input data type of model "{config.build_name}" changed from '
                f"{state.expected_input_dtypes} to {input_dtypes} "
                f"since the last time it was built."
            )
            result.append(msg)

        if len(changed_args) > 0:
            for key_name, current_arg, previous_arg in changed_args:
                msg = (
                    f'build_model() argument "{key_name}" for build '
                    f"{config.build_name} changed from "
                    f"{previous_arg} to {current_arg} since the last build."
                )
                result.append(msg)
    else:

        if (
            state.build_status == build.Status.FAILED_BUILD
            or state.build_status == build.Status.BUILD_RUNNING
        ) and onnxflow_version == state.onnxflow_version:
            msg = (
                "build_model() has detected that you already attempted building "
                "this model with the exact same model, inputs, options, and version of "
                "onnxflow, and that build failed."
            )
            result.append(msg)

    return result


def _begin_fresh_build(
    state_args: Dict,
    state_type: Type = build.State,
) -> build.State:
    # Wipe this model's directory in the cache and start with a fresh State.
    cache.rmdir(
        build.output_dir(state_args["cache_dir"], state_args["config"].build_name)
    )
    state = state_type(**state_args)
    state.save()

    return state


def _rebuild_if_needed(
    problem_report: str, state_args: Dict, state_type: Type = build.State
):
    build_name = state_args["config"].build_name
    msg = (
        f"build_model() discovered a cached build of {build_name}, but decided to "
        "rebuild for the following reasons: \n\n"
        f"{problem_report} \n\n"
        "build_model() will now rebuild your model to ensure correctness. You can change this "
        "policy by setting the build_model(rebuild=...) argument."
    )
    printing.log_warning(msg)

    return _begin_fresh_build(state_args, state_type=state_type)


def load_or_make_state(
    config: build.Config,
    cache_dir: str,
    rebuild: str,
    model_type: build.ModelType,
    monitor: bool,
    model: build.UnionValidModelInstanceTypes = None,
    inputs: Optional[Dict[str, Any]] = None,
    quantization_samples: Optional[Collection] = None,
    state_type: Type = build.State,
    cache_validation_func: Callable = validate_cached_model,
    extra_state_args: Optional[Dict] = None,
) -> build.State:
    """
    Decide whether we can load the model from the model cache
    (return a valid State instance) or whether we need to rebuild it (return
    a new State instance).
    """

    # Put all the args for making a new State instance into a dict
    # to help the following code be cleaner
    state_args = {
        "model": model,
        "inputs": inputs,
        "monitor": monitor,
        "rebuild": rebuild,
        "cache_dir": cache_dir,
        "config": config,
        "model_type": model_type,
        "quantization_samples": quantization_samples,
    }

    # Allow customizations of onnxflow to supply additional args
    if extra_state_args is not None:
        state_args.update(extra_state_args)

    if rebuild == "always":
        return _begin_fresh_build(state_args, state_type)
    else:
        # Try to load state and check if model successfully built before
        if os.path.isfile(build.state_file(cache_dir, config.build_name)):
            try:
                state = build.load_state(
                    cache_dir,
                    config.build_name,
                    state_type=state_type,
                )

                # if the previous build is using quantization while the current is not
                # or vice versa
                if state.quantization_samples and quantization_samples is None:
                    if rebuild == "never":
                        msg = (
                            f"Model {config.build_name} was built in a previous call to "
                            "build_model() with post-training quantization sample enabled."
                            "However, post-training quantization is not enabled in the "
                            "current build. Rebuild is necessary but currently the rebuild"
                            "policy is set to 'never'. "
                        )
                        raise exp.CacheError(msg)

                    msg = (
                        f"Model {config.build_name} was built in a previous call to "
                        "build_model() with post-training quantization sample enabled."
                        "However, post-training quantization is not enabled in the "
                        "current build. Starting a fresh build."
                    )

                    printing.log_info(msg)
                    return _begin_fresh_build(state_args, state_type)

                if not state.quantization_samples and quantization_samples is not None:
                    if rebuild == "never":
                        msg = (
                            f"Model {config.build_name} was built in a previous call to "
                            "build_model() with post-training quantization sample disabled."
                            "However, post-training quantization is enabled in the "
                            "current build. Rebuild is necessary but currently the rebuild"
                            "policy is set to 'never'. "
                        )
                        raise exp.CacheError(msg)

                    msg = (
                        f"Model {config.build_name} was built in a previous call to "
                        "build_model() with post-training quantization sample disabled."
                        "However, post-training quantization is enabled in the "
                        "current build. Starting a fresh build."
                    )

                    printing.log_info(msg)
                    return _begin_fresh_build(state_args, state_type)

            except exp.StateError as e:
                problem = (
                    "- build_model() failed to load "
                    f"{build.state_file(cache_dir, config.build_name)}"
                )

                if rebuild == "if_needed":
                    return _rebuild_if_needed(problem, state_args, state_type)
                else:
                    # Give the rebuild="never" users a chance to address the problem
                    raise exp.CacheError(e)

            if (
                model_type == build.ModelType.UNKNOWN
                and state.build_status == build.Status.SUCCESSFUL_BUILD
            ):
                msg = (
                    "Model caching is disabled for successful builds against custom Sequences. "
                    "Your model will rebuild whenever you call build_model() on it."
                )
                printing.log_warning(msg)

                return _begin_fresh_build(state_args, state_type)
            elif (
                model_type == build.ModelType.UNKNOWN
                and state.build_status == build.Status.PARTIAL_BUILD
            ):
                msg = (
                    f"Model {config.build_name} was partially built in a previous call to "
                    "build_model(). This call to build_model() found that partial build and "
                    "is loading it from the build cache."
                )

                printing.log_info(msg)
            else:
                cache_problems = cache_validation_func(
                    config=config,
                    model_type=model_type,
                    state=state,
                    model=model,
                    inputs=inputs,
                )

                if len(cache_problems) > 0:
                    cache_problems = [f"- {msg}" for msg in cache_problems]
                    problem_report = "\n".join(cache_problems)

                    if rebuild == "if_needed":
                        return _rebuild_if_needed(
                            problem_report, state_args, state_type
                        )
                    if rebuild == "never":
                        msg = (
                            "build_model() discovered a cached build of "
                            f"{config.build_name}, and found that it "
                            "is likely invalid for the following reasons: \n\n"
                            f"{problem_report} \n\n"
                            'However, since you have set rebuild="never", build_model() will '
                            "attempt to load the build from cache anyways (with no guarantee of "
                            "functionality or correctness). "
                        )
                        printing.log_warning(msg)

            # Ensure the model and inputs are part of the state
            # This is useful  when loading models that still need to be built
            state.save_when_setting_attribute = False
            if state.model is None:
                state.model = model
            if state.inputs is None:
                state.inputs = inputs
            state.save_when_setting_attribute = True

            return state

        else:
            # No state file found, so we have to build
            return _begin_fresh_build(state_args, state_type)


model_type_to_sequence = {
    build.ModelType.PYTORCH: default_pytorch_sequence,
    build.ModelType.KERAS: default_keras_sequence,
    build.ModelType.ONNX_FILE: default_onnx_sequence,
    build.ModelType.HUMMINGBIRD: default_hummingbird_sequence,
}

model_type_to_export_sequence = {
    build.ModelType.PYTORCH: default_pytorch_export_sequence,
    build.ModelType.KERAS: default_keras_export_sequence,
    build.ModelType.ONNX_FILE: default_onnx_export_sequence,
    build.ModelType.HUMMINGBIRD: default_hummingbird_export_sequence,
}

model_type_to_sequence_with_quantization = {
    build.ModelType.PYTORCH: pytorch_sequence_with_quantization,
}


def validate_inputs(inputs: Dict):
    """
    Check the model's inputs and make sure they are legal. Raise an exception
    if they are not legal.
    TODO: it may be wise to validate the inputs against the model, or at least
    the type of model, as well.
    """

    if inputs is None:
        msg = """
        build_model() requires model inputs. Check your call to build_model() to make sure
        you are passing the inputs argument.
        """
        raise exp.IntakeError(msg)

    if not isinstance(inputs, dict):
        msg = f"""
        The "inputs" argument to build_model() is required to be a dictionary, where the
        keys map to the named arguments in the model's forward function. The inputs
        received by build_model() were of type {type(inputs)}, not dict.
        """
        raise exp.IntakeError(msg)


def identify_model_type(model) -> build.ModelType:
    # Validate that the model's type is supported by build_model()
    # and assign a ModelType tag
    if isinstance(model, (torch.nn.Module, torch.jit.ScriptModule)):
        model_type = build.ModelType.PYTORCH
    elif isinstance(model, str):
        if model.endswith(".onnx"):
            model_type = build.ModelType.ONNX_FILE
    elif tf_helpers.is_keras_model(model):
        model_type = build.ModelType.KERAS
        if not tf_helpers.is_executing_eagerly():
            raise exp.IntakeError(
                "`build_model()` requires Keras models to be run in eager execution mode. "
                "Enable eager execution to continue."
            )
        if not model.built:
            raise exp.IntakeError(
                "Keras model has not been built. Please call "
                "model.build(input_shape) before running build_model()"
            )
    elif hummingbird.is_supported_model(model):
        model_type = build.ModelType.HUMMINGBIRD
    else:
        raise exp.IntakeError(
            "Argument 'model' passed to build_model() is "
            f"of unsupported type {type(model)}"
        )

    return model_type


def model_intake(
    user_model,
    user_inputs,
    user_sequence: Optional[stage.Sequence],
    user_quantization_samples: Optional[Collection] = None,
    override_quantization_sequence_map: Optional[
        Dict[build.ModelType, stage.Sequence]
    ] = None,
    override_sequence_map: Dict[build.ModelType, stage.Sequence] = None,
    export_only: bool = False,
) -> Tuple[Any, Any, stage.Sequence, build.ModelType, str]:

    # Model intake structure options:
    # user_model
    #    |
    #    |------- path to onnx model file
    #    |
    #    |------- pytorch model object
    #    |
    #    |------- keras model object
    #    |
    #    |------- Hummingbird-supported model object

    if override_quantization_sequence_map is None:
        quantization_sequence_map = model_type_to_sequence_with_quantization
    else:
        quantization_sequence_map = override_quantization_sequence_map

    if override_sequence_map is None:
        if export_only:
            sequence_map = model_type_to_export_sequence
        else:
            sequence_map = model_type_to_sequence
    else:
        sequence_map = override_sequence_map

    if user_sequence is None or user_sequence.enable_model_validation:

        if user_model is None and user_inputs is None:
            msg = """
            You are running build_model() without any model, inputs, or custom Sequence. The purpose
            of non-customized build_model() is to build a model against some inputs, so you need to
            provide both.
            """
            raise exp.IntakeError(msg)

        # Make sure that if the model is a file path, it is valid
        if isinstance(user_model, str):
            if not os.path.isfile(user_model):
                msg = f"""
                build_model() model argument was passed a string (path to a model file),
                however no file was found at {user_model}.
                """
                raise exp.IntakeError(msg)

            if not user_model.endswith(".onnx"):
                msg = f"""
                build_model() received a model argument that was a string. However, model string
                arguments are required to be a path to a .onnx file, but the argument was: {user_model}
                """
                raise exp.IntakeError(msg)

            # Create dummy inputs based on the ONNX spec, if none were provided by the user
            if user_inputs is None:
                inputs = onnx_helpers.dummy_inputs(user_model)
            else:
                inputs = user_inputs
        else:
            inputs = user_inputs

        model_type = identify_model_type(user_model)

        sequence = user_sequence
        if sequence is None:
            # Assign a sequence based on the ModelType
            if user_quantization_samples:
                if model_type != build.ModelType.PYTORCH:
                    raise exp.IntakeError(
                        "Currently, post training quantization only supports Pytorch models."
                    )
                sequence = quantization_sequence_map[model_type]
            else:
                sequence = sequence_map[model_type]

        validate_inputs(inputs)

    else:
        # We turn off a significant amount of automation and validation
        # to provide custom stages and sequences with maximum flexibility
        inputs = user_inputs
        sequence = user_sequence
        model_type = build.ModelType.UNKNOWN

    return (user_model, inputs, sequence, model_type)
