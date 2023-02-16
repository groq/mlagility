import sys
import os
import inspect
import importlib.util
import copy
import time
import shlex
import functools
import dataclasses
import pathlib
import traceback
from typing import Union, List, Dict
from types import FrameType, TracebackType
from enum import Enum
import torch
from groqflow.common import printing
import groqflow.common.build as build
import groqflow.common.exceptions as exp
from groqflow.justgroqit.stage import Sequence
import mlagility.analysis.status as status
import mlagility.analysis.util as util
import mlagility.analysis.tf_helpers as tf_helpers
import mlagility.common.labels as labels
from mlagility.api.model_api import benchmark_model
import mlagility.common.filesystem as filesystem


class Action(Enum):
    ANALYZE = "analyze"
    BUILD = "build"
    BENCHMARK = "benchmark"


@dataclasses.dataclass
class TracerArgs:
    input: str
    device: str
    backend: str
    actions: List[Action]
    lean_cache: bool
    targets: List[str]
    max_depth: int
    cache_dir: str
    rebuild: str
    groq_compiler_flags: List[str]
    groq_assembler_flags: List[str]
    groq_num_chips: int
    groqview: bool
    models_found: Dict[str, util.ModelInfo] = dataclasses.field(default_factory=dict)
    script_name: str = None
    sequence: Sequence = None

    @functools.cached_property
    def labels(self) -> Dict[str, str]:
        return labels.load_from_file(self.input)

    @functools.cached_property
    def torch_activations(self) -> List[str]:
        act = util.get_classes(torch.nn.modules.activation)
        act += tf_helpers.get_transformers_activations()
        return act

def _store_traceback(model_info: util.ModelInfo):
    """
    Store the traceback from an exception into model_info so that
    we can print it during the status update.
    """
    
    exc_type, exc_value, exc_traceback = sys.exc_info()
    model_info.traceback = traceback.format_exception(
        exc_type, exc_value, exc_traceback
    )

def call_benchit(
    model_inputs: dict, model_info: util.ModelInfo, tracer_args: TracerArgs
) -> None:
    """
    Calls the benchit function from within the model forward function
    """

    # Update status to "computing"
    model_info.status_message = "Computing..."
    model_info.status_message_color = printing.Colors.OKBLUE
    status.update(tracer_args.models_found)

    # Get a copy of the keyword arguments
    args, kwargs = model_inputs
    inputs = {}
    for k in kwargs.keys():
        if torch.is_tensor(kwargs[k]):
            inputs[k] = torch.tensor(kwargs[k].detach().numpy())
        else:
            inputs[k] = copy.deepcopy(kwargs[k])

    # Convert all positional arguments into keyword arguments
    if args != ():
        if model_info.model_type == build.ModelType.PYTORCH:
            forward_function = model_info.model.forward
        elif model_info.model_type == build.ModelType.KERAS:
            forward_function = model_info.model.call
        all_args = list(inspect.signature(forward_function).parameters.keys())
        for i in range(len(args)):
            if torch.is_tensor(args[i]):
                inputs[all_args[i]] = torch.tensor(args[i].detach().numpy())
            else:
                inputs[all_args[i]] = args[i]
    model_info.inputs = inputs

    cache_dir = (
        filesystem.DEFAULT_CACHE_DIR
        if tracer_args.cache_dir is None
        else tracer_args.cache_dir
    )
    build_name = f"{tracer_args.script_name}_{model_info.hash}"

    # Save model labels
    tracer_args.labels["class"] = [f"{type(model_info.model).__name__}"]
    labels.save_to_cache(cache_dir, build_name, tracer_args.labels)

    try:
        perf = benchmark_model(
            model_info.model,
            inputs,
            device=tracer_args.device,
            backend=tracer_args.backend,
            build_name=build_name,
            cache_dir=cache_dir,
            build_only=Action.BENCHMARK not in tracer_args.actions,
            lean_cache=tracer_args.lean_cache,
            groq_num_chips=tracer_args.groq_num_chips,
            groq_compiler_flags=tracer_args.groq_compiler_flags,
            groq_assembler_flags=tracer_args.groq_assembler_flags,
            groqview=tracer_args.groqview,
            sequence=tracer_args.sequence,
        )

        if Action.BENCHMARK in tracer_args.actions:
            model_info.status_message = "Model successfully benchmarked!"
            model_info.performance = perf
        else:
            model_info.status_message = "Model successfully built!"
        model_info.status_message_color = printing.Colors.OKGREEN

    except exp.GroqitStageError:
        load_state = build.load_state(build_name=build_name)
        if len(load_state.info.opt_onnx_unsupported_ops) > 0:
            model_info.status_message = "Unsupported op(s) " + ", ".join(
                load_state.info.opt_onnx_unsupported_ops
            )
        else:
            model_info.status_message = "Build Error: see log files for details."
        model_info.status_message_color = printing.Colors.WARNING
        
        _store_traceback(model_info)

    except exp.GroqFlowError:
        model_info.status_message = "GroqFlowError: see log files for details."
        model_info.status_message_color = printing.Colors.WARNING

        _store_traceback(model_info)

    # This broad exception is ok since enumerating all exceptions is
    # not possible, as the tested software continuously evolves.
    except Exception as e:  # pylint: disable=broad-except
        util.stop_stdout_forward()
        model_info.status_message = f"Unknown benchit error: {e}"
        model_info.status_message_color = printing.Colors.WARNING
        
        _store_traceback(model_info)


def get_model_hash(
    model: Union[torch.nn.Module, "tf.keras.Model"], model_type: build.ModelType
):
    return build.hash_model(model, model_type, hash_params=False)[:8]


def store_model_info(
    model: Union[torch.nn.Module, "tf.keras.Model"],
    model_name: str,
    model_type: build.ModelType,
    frame: FrameType,
    event: str,
    tracer_args: TracerArgs,
    depth: int,
    parent_hash: str,
):
    # Getting the model hash is only possible after the first inference of Keras models
    model_hash = get_model_hash(model, model_type)

    # File where the model was found
    file = str(frame)[str(frame).find("file ") + 6 : str(frame).find("',")]

    # Line where the model was found
    line = frame.f_lineno if event == "return" else frame.f_lineno - 1

    # Keep track of all models details

    # If we have already found a model, don't add it to models_found again
    # We have to use both the model hash and the script name, since we don't
    # want to ignore a model if it was explicitly called in two different scripts
    identifier = f"{model_hash}_{tracer_args.script_name}"
    model_already_found = False
    for model_info in tracer_args.models_found.values():
        if identifier == f"{model_info.hash}_{model_info.script_name}":
            model_already_found = True

    if not model_already_found:
        tracer_args.models_found[model_hash] = util.ModelInfo(
            model=model,
            name=model_name,
            file=file,
            line=line,
            depth=depth,
            hash=model_hash,
            parent_hash=parent_hash,
            is_target=model_hash in tracer_args.targets or tracer_args.targets == [],
            build_model=Action.BUILD in tracer_args.actions,
            model_type=model_type,
            script_name=tracer_args.script_name,
        )


def explore_frame(
    frame,
    event,
    local_var_name,
    local_var,
    tracer_args: TracerArgs,
    depth: int = 0,
    parent_hash: Union[str, None] = None,
):
    """
    This function checks whether local_var is a torch or keras model.
    If it is, we will modify its forward function to know when it
    is called.
    """

    # Skip all variables that are not a subclass of torch.nn.Module/tf.keras.Model
    # Note: try block used since dead weakreferences fail when checking subclass
    try:
        if issubclass(type(local_var), torch.nn.Module):
            if type(local_var) in tracer_args.torch_activations:
                return
            model_type = build.ModelType.PYTORCH
        elif tf_helpers.is_keras_subclass(type(local_var)):
            model_type = build.ModelType.KERAS
        else:
            return
    except AttributeError:
        return

    # Skip self variable and variable names commonly used by child models
    if (
        local_var_name == "self"
        or local_var_name == "instance"
        or local_var_name == "child"
        or local_var_name == "layer"
        or local_var_name == "module"
    ):
        return

    # Check if we are inside of a subclass of torch.nn.Module or tf.keras.model
    inside_class = False
    inside_nn_subclass = False
    if "self" in frame.f_locals:
        self_var = frame.f_locals["self"]
        inside_class = type(self_var)
        inside_nn_subclass = issubclass(
            inside_class, torch.nn.Module
        ) or tf_helpers.is_keras_subclass(inside_class)

    if not hasattr(local_var, "forward_instrumented") and not inside_nn_subclass:

        if model_type == build.ModelType.PYTORCH:

            # Avoid instrumenting models before they have been fully loaded
            if util.count_parameters(local_var, model_type) == 0:
                return

            # Mark this model as instrumented
            local_var.forward_instrumented = True

            # Create a copy of the old forward function
            old_forward = local_var.forward

            # Recursively look for sub-models within the found model
            # This is only possible on Pytorch, since each layer of a torch.nn.module
            # is also a torch.nn.module.
            model_hash = get_model_hash(local_var, model_type)
            if depth < tracer_args.max_depth:
                recursive_search(
                    frame, event, local_var, depth, model_hash, tracer_args
                )

            # We can keep track of Pytorch models even before they are executed
            store_model_info(
                local_var,
                local_var_name,
                model_type,
                frame,
                event,
                tracer_args,
                depth,
                parent_hash,
            )
        elif model_type == build.ModelType.KERAS:
            # Mark this model as instrumented
            local_var.forward_instrumented = True

            # Create a copy of the old forward function
            old_forward = local_var.call

            # Raise exception if user tries to use max_depth!=0 for a keras model
            if tracer_args.max_depth != 0:
                raise exp.GroqFlowError("max_depth is not supported for Keras models")
        local_var.old_forward = old_forward

        def forward_spy(*args, **kwargs):

            tracer = sys.getprofile()
            if tracer is not None:
                # Turn tracing off while the model is being executed for speed
                sys.setprofile(None)
            elif depth == 0:
                # If this model is being executed and the tracing is already off
                # we are calling a module within a parent module. We only run
                # groqit on child models if the user has explicitly asked us to
                # do so by setting the max_depth flag.
                return old_forward(*args, **kwargs)

            # Keep track of execution time
            start_time = time.time()
            outputs = old_forward(*args, **kwargs)
            end_time = time.time()

            # We can only keep track of keras models once they have been executed
            if model_type == build.ModelType.KERAS:
                store_model_info(
                    local_var,
                    local_var_name,
                    model_type,
                    frame,
                    event,
                    tracer_args,
                    depth,
                    parent_hash,
                )
            model_hash = get_model_hash(local_var, model_type)
            model_info = tracer_args.models_found[model_hash]
            model_info.exec_time = model_info.exec_time + end_time - start_time

            model_info.executed = model_info.executed + 1

            # Call groqit if this is the first time the model is being executed
            # and this model has been selected by the user
            if (
                model_info.executed == 1
                and model_info.is_target
                and (model_info.build_model)
            ):
                call_benchit(
                    model_inputs=[args, kwargs],
                    model_info=model_info,
                    tracer_args=tracer_args,
                )
                # Ensure that groqit() doesn't interfere with our execution count
                model_info.executed = 1

            status.update(tracer_args.models_found)

            # Turn tracing on again after computing the outputs
            sys.setprofile(tracer)

            return outputs

        # The inspect module offers the ability to actually copy the signature of the wrapped
        # function. This allows other functions to see the correct parameters instead of the
        # enigmatic *args, **kwargs. This is especially important for Keras, since it heavily
        # relies on inspections to the call function.
        forward_spy.__signature__ = inspect.signature(old_forward)

        # Use modified forward/call function
        if model_type == build.ModelType.PYTORCH:
            local_var.forward = forward_spy
        elif model_type == build.ModelType.KERAS:
            local_var.call = forward_spy


def tracefunc(
    frame: FrameType, event: str, _, tracer_args: TracerArgs
) -> TracebackType:
    """
    This function is used to trace the program as it runs in order
    to keep track of all all instantiated models.
    This function is passed to sys.setprofile() as a callback function.
    It receives three arguments:
        frame (the stack frame from the code being run),
        event (a string naming the type of notification), and
        arg (an event-specific value)

    """

    # Create a copy of f_locals.keys() to avoid errors due to dict changing
    local_names = list(frame.f_locals.keys())

    # Loop over all local variables to check if new models can be found
    for local_var_name in local_names:
        explore_frame(
            frame,
            event,
            local_var_name,
            frame.f_locals[local_var_name],
            tracer_args=tracer_args,
            depth=0,
        )

    return tracefunc


def recursive_search(
    frame: FrameType,
    event: str,
    model: Union[torch.nn.Module, "tf.keras.Model"],
    depth: int,
    parent_hash: Union[str, None],
    tracer_args: TracerArgs,
):
    """
    Recursively check for submodels within found models
    """
    element_names = list(dict(model.named_modules()).keys())[1:]
    for element_name in element_names:
        if hasattr(model, element_name):
            element = getattr(model, element_name)
            if issubclass(type(element), torch.nn.Module):
                explore_frame(
                    frame,
                    event,
                    element_name,
                    element,
                    tracer_args,
                    depth=depth + 1,
                    parent_hash=parent_hash,
                )


def evaluate_script(
    tracer_args: TracerArgs, input_args: str = None
) -> Dict[str, util.ModelInfo]:
    # Trim the ".py"
    tracer_args.script_name = pathlib.Path(tracer_args.input).stem

    # Get a pointer to the script's python module
    spec = importlib.util.spec_from_file_location(
        tracer_args.script_name, tracer_args.input
    )
    module = importlib.util.module_from_spec(spec)

    # Overwriting argv to import input script using "input-args"
    if input_args is None:
        input_args = []
    else:
        input_args = shlex.split(input_args[0])
    sys.argv = [tracer_args.input] + input_args
    sys.path.append(os.getcwd())

    # Create a tracer object that bundles a callback function with some args
    tracer = functools.partial(tracefunc, tracer_args=tracer_args)

    # Enabling autogroq via setprofile
    sys.setprofile(tracer)

    # Import input script. Each executed frame of the input script will
    # trigger the tracefunc() callback function (defined above)
    spec.loader.exec_module(module)

    # Stop profiling when we're done executing the module
    sys.setprofile(None)

    return tracer_args.models_found
