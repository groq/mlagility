import os
from typing import Optional, List, Dict, Any
from collections.abc import Collection
import onnxflow.model as omodel
import onnxflow.justbuildit.ignition as ignition
import onnxflow.justbuildit.stage as stage
import onnxflow.common.printing as printing
import onnxflow.common.build as build


def build_model(
    model: build.UnionValidModelInstanceTypes = None,
    inputs: Optional[Dict[str, Any]] = None,
    build_name: Optional[str] = None,
    cache_dir: str = build.DEFAULT_CACHE_DIR,
    monitor: bool = True,
    rebuild: Optional[str] = None,
    sequence: Optional[List[stage.Stage]] = None,
    quantization_samples: Collection = None,
    onnx_opset: Optional[int] = None,
    export_only: bool = False,
) -> omodel.BaseModel:

    """Use build a model instance into an optimized ONNX file.

    Args:
        model: Model to be mapped to an optimized ONNX file, which can be a PyTorch
            model instance, Keras model instance, Hummingbird model instance,
            or a path to an ONNX file.
        inputs: Example inputs to the user's model. The ONNX file will be
            built to handle inputs with the same static shape only.
        build_name: Unique name for the model that will be
            used to store the ONNX file and build state on disk. Defaults to the
            name of the file that calls build_model().
        cache_dir: Directory to use as the cache for this build. Output files
            from this build will be stored at cache_dir/build_name/
            Defaults to the current working directory, but we recommend setting it to
            an absolute path of your choosing.
        monitor: Display a monitor on the command line that
            tracks the progress of this function as it builds the ONNX file.
        rebuild: determines whether to rebuild or load a cached build. Options:
            - "if_needed" (default): overwrite invalid cached builds with a rebuild
            - "always": overwrite valid cached builds with a rebuild
            - "never": load cached builds without checking validity, with no guarantee
                of functionality or correctness
            - None: Falls back to default
        sequence: Override the default sequence of build stages. Power
            users only.
        quantization_samples: If set, performs post-training quantization
            on the ONNX model using the provided samplesIf the previous build used samples
            that are different to the samples used in current build, the "rebuild"
            argument needs to be manually set to "always" in the current build
            in order to create a new ONNX file.
        onnx_opset: ONNX opset to use during ONNX export.
        export_only: Export the model to ONNX but do not apply any optimizations to the
            resulting ONNX file.
    """

    # Support "~" in the cache_dir argument
    parsed_cache_dir = os.path.expanduser(cache_dir)

    # Validate and lock in the config (user arguments that
    # configure the build) that will be used by the rest of the toolchain
    config = ignition.lock_config(
        model=model,
        build_name=build_name,
        sequence=sequence,
        onnx_opset=onnx_opset,
    )

    # Analyze the user's model argument and lock in the model, inputs,
    # and sequence that will be used by the rest of the toolchain
    (model_locked, inputs_locked, sequence_locked, model_type,) = ignition.model_intake(
        model,
        inputs,
        sequence,
        user_quantization_samples=quantization_samples,
        export_only=export_only,
    )

    # Get the state of the model from the cache if a valid build is available
    state = ignition.load_or_make_state(
        config=config,
        cache_dir=parsed_cache_dir,
        rebuild=rebuild or build.DEFAULT_REBUILD_POLICY,
        model_type=model_type,
        monitor=monitor,
        model=model_locked,
        inputs=inputs_locked,
        quantization_samples=quantization_samples,
    )

    # Return a cached build if possible, otherwise prepare the model State for
    # a build
    if state.build_status == build.Status.SUCCESSFUL_BUILD:
        # Successful builds can be loaded from cache and returned with
        # no additional steps
        additional_msg = " (build_name auto-selected)" if config.auto_name else ""
        printing.log_success(
            f' Build "{config.build_name}"{additional_msg} found in cache. Loading it!',
        )

        return omodel.load(config.build_name, state.cache_dir)

    state.quantization_samples = quantization_samples

    sequence_locked.show_monitor(config, state.monitor)
    state = sequence_locked.launch(state)

    if state.build_status == build.Status.SUCCESSFUL_BUILD:
        printing.log_success(
            f"\n    Saved to **{build.output_dir(state.cache_dir, config.build_name)}**"
        )

        return omodel.load(config.build_name, state.cache_dir)

    else:
        printing.log_success(
            f"Build Sequence {sequence_locked.unique_name} completed successfully"
        )
        msg = """
        build_model() only returns a Model instance if the Sequence includes a Stage
        that sets state.build_status=onnxflow.build.Status.SUCCESSFUL_BUILD.
        """
        printing.log_warning(msg)
