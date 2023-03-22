from typing import Optional, List, Tuple, Union, Dict, Any
from collections.abc import Collection
import sys
import os
import torch
import onnxflow.common.build as build
import onnxflow.common.cache as cache
import onnxflow.common.exceptions as exp
import onnxflow.common.printing as printing
import onnxflow.common.tf_helpers as tf_helpers
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

default_pytorch_sequence = stage.Sequence(
    "default_pytorch_export_sequence",
    "Exporting PyTorch Model",
    [export.ExportPytorchModel(), polish_onnx_sequence],
)

pytorch_sequence_with_quantization = stage.Sequence(
    "pytorch_export_sequence_with_quantization",
    "Exporting PyTorch Model and Quantizating Exported ONNX",
    [
        export.ExportPytorchModel(),
        export.OptimizeOnnxModel(),
        export.QuantizeONNXModel(),
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


default_onnx_sequence = stage.Sequence(
    "default_onnx_sequence",
    "Building ONNX Model",
    [
        export.ReceiveOnnxModel(),
        polish_onnx_sequence,
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
    build_name: Optional[str] = None,
    sequence: stage.Sequence = None,
) -> Tuple[build.Config, bool]:

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

    # Store the args that should be immutable
    config = build.Config(
        build_name=build_name,
        sequence=stage_names,
    )

    return config, auto_name


def _validate_cached_model(
    config: build.Config,
    model_type: build.ModelType,
    state: build.State,
    version: str,
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

    current_version_decoded = _decode_version_number(version)
    state_version_decoded = _decode_version_number(state.onnxflow_version)

    out_of_date: Union[str, bool] = False
    if current_version_decoded["major"] > state_version_decoded["major"]:
        out_of_date = "major"
    elif current_version_decoded["minor"] > state_version_decoded["minor"]:
        out_of_date = "minor"

    if out_of_date:
        msg = (
            f"Your build {state.config.build_name} was previously built against "
            f"onnxflow version {state.onnxflow_version}, "
            f"however you are now using onxxflow version {version}. The previous build is "
            f"incompatible with this version of onnxflow, as indicated by the {out_of_date} "
            "version number changing. See **docs/versioning.md** for details."
        )
        result.append(msg)

    if model is not None:
        model_changed = state.model_hash != build.hash_model(model, model_type)
    else:
        model_changed = False

    if inputs is not None:
        input_shapes, input_dtypes = build.get_shapes_and_dtypes(inputs)
        input_shapes_changed = state.expected_input_shapes != input_shapes
        input_dtypes_changed = state.expected_input_dtypes != input_dtypes
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
            msg = (
                f'Input shape of model "{config.build_name}" changed from '
                f"{state.expected_input_shapes} to {input_shapes} "
                f"since the last time it was built."
            )
            result.append(msg)

        if input_dtypes_changed:
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
        ) and version == state.onnxflow_version:
            msg = (
                "build_model() has detected that you already attempted building this model with the "
                "exact same model, inputs, options, and version of onnxflow, and that build failed."
            )
            result.append(msg)

    return result


def _decode_version_number(version: str) -> Dict[str, int]:
    numbers = [int(x) for x in version.split(".")]
    return {"major": numbers[0], "minor": numbers[1], "patch": numbers[0]}


def _begin_fresh_build(
    model: build.UnionValidModelInstanceTypes,
    inputs: Optional[Dict[str, Any]],
    monitor: bool,
    rebuild: str,
    cache_dir: str,
    config: build.Config,
    model_type: build.ModelType,
    version: str,
    quantization_samples: Collection,
) -> build.State:
    # Wipe this model's directory in the cache and start with a fresh State.
    cache.rmdir(build.output_dir(cache_dir, config.build_name))
    state = build.State(
        model=model,
        inputs=inputs,
        monitor=monitor,
        rebuild=rebuild,
        cache_dir=cache_dir,
        config=config,
        model_type=model_type,
        onnxflow_version=version,
        quantization_samples=quantization_samples,
    )
    state.save()

    return state


def _rebuild_if_needed(problem_report: str, state_args: Dict):
    build_name = state_args["config"].build_name
    msg = (
        f"build_model() discovered a cached build of {build_name}, but decided to "
        "rebuild for the following reasons: \n\n"
        f"{problem_report} \n\n"
        "build_model() will now rebuild your model to ensure correctness. You can change this "
        "policy by setting the build_model(rebuild=...) argument."
    )
    printing.log_warning(msg)

    return _begin_fresh_build(**state_args)


def load_or_make_state(
    config: build.Config,
    cache_dir: str,
    rebuild: str,
    model_type: build.ModelType,
    monitor: bool,
    model: build.UnionValidModelInstanceTypes = None,
    inputs: Optional[Dict[str, Any]] = None,
    quantization_samples: Optional[Collection] = None,
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
        "version": onnxflow_version,
        "quantization_samples": quantization_samples,
    }

    if rebuild == "always":
        return _begin_fresh_build(**state_args)
    else:
        # Try to load state and check if model successfully built before
        if os.path.isfile(build.state_file(cache_dir, config.build_name)):
            try:
                state = build.load_state(cache_dir, config.build_name)

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
                    return _begin_fresh_build(**state_args)

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
                    return _begin_fresh_build(**state_args)

            except exp.StateError as e:
                problem = (
                    "- build_model() failed to load "
                    f"{build.state_file(cache_dir, config.build_name)}"
                )

                if rebuild == "if_needed":
                    return _rebuild_if_needed(problem, state_args)
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

                return _begin_fresh_build(**state_args)
            elif (
                model_type == build.ModelType.UNKNOWN
                and state.build_status == build.Status.PARTIAL_BUILD
            ):
                msg = (
                    f"Model {config.build_name} was partially built in a previous call to "
                    "build_model(). This call to build_model() found that partial build and is loading "
                    "it from the model cache."
                )

                printing.log_info(msg)
                return state
            else:
                cache_problems = _validate_cached_model(
                    config=config,
                    model_type=model_type,
                    state=state,
                    version=onnxflow_version,
                    model=model,
                    inputs=inputs,
                )

                if len(cache_problems) > 0:
                    cache_problems = [f"- {msg}" for msg in cache_problems]
                    problem_report = "\n".join(cache_problems)

                    if rebuild == "if_needed":
                        return _rebuild_if_needed(problem_report, state_args)
                    if rebuild == "never":
                        msg = (
                            "build_model() discovered a cached build of "
                            f"{config.build_name}, and found that it "
                            "is likely invalid for the following reasons: \n\n"
                            f"{problem_report} \n\n"
                            'However, since you have set rebuild="never", build_model() will attempt '
                            "to load the build from cache anyways (with no guarantee of "
                            "functionality or correctness). "
                        )
                        printing.log_warning(msg)
                        return state
                else:
                    return state

        else:
            # No state file found, so we have to build
            return _begin_fresh_build(**state_args)


def _load_model_from_file(path_to_model, user_inputs):
    if not os.path.isfile(path_to_model):
        msg = f"""
        build_model() model argument was passed a string (path to a model file),
        however no file was found at {path_to_model}.
        """
        raise exp.IntakeError(msg)

    if path_to_model.endswith(".onnx"):
        return path_to_model, user_inputs

    else:
        msg = f"""
        build_model() received a model argument that was a string. However, model string
        arguments are required to be a path to either a .py or .onnx file, and the
        following argument is neither: {path_to_model}
        """
        raise exp.IntakeError(msg)


model_type_to_sequence = {
    build.ModelType.PYTORCH: default_pytorch_sequence,
    build.ModelType.KERAS: default_keras_sequence,
    build.ModelType.ONNX_FILE: default_onnx_sequence,
    build.ModelType.HUMMINGBIRD: default_hummingbird_sequence,
}

model_type_to_sequence_with_quantization = {
    build.ModelType.PYTORCH: pytorch_sequence_with_quantization,
}


def _validate_inputs(inputs: Dict):
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

    if user_sequence is None or user_sequence.enable_model_validation:

        if user_model is None and user_inputs is None:
            msg = """
            You are running build_model() without any model, inputs, or custom Sequence. The purpose
            of non-customized build_model() is to build a model against some inputs, so you need to
            provide both.
            """
            raise exp.IntakeError(msg)

        # Convert paths to models into models
        if isinstance(user_model, str):
            model, inputs = _load_model_from_file(user_model, user_inputs)
        else:
            model, inputs = user_model, user_inputs

        model_type = identify_model_type(model)

        sequence = user_sequence
        if sequence is None:
            # Assign a sequence based on the ModelType
            if user_quantization_samples:
                if model_type != build.ModelType.PYTORCH:
                    raise exp.IntakeError(
                        "Currently, post training quantization only supports Pytorch models."
                    )
                sequence = model_type_to_sequence_with_quantization[model_type]
            else:
                sequence = model_type_to_sequence[model_type]

        _validate_inputs(inputs)

    else:
        # We turn off a significant amount of automation and validation
        # to provide custom stages and sequences with maximum flexibility
        sequence = user_sequence
        model = user_model
        inputs = user_inputs
        model_type = build.ModelType.UNKNOWN

    return (model, inputs, sequence, model_type)
