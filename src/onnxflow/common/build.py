import os
import sys
import pathlib
import copy
import enum
from typing import Optional, Any, List, Dict, Union, Type
from collections.abc import Collection
import dataclasses
import hashlib
import psutil
import yaml
import torch
import numpy as np
import sklearn.base
import onnxflow.common.exceptions as exp
import onnxflow.common.tf_helpers as tf_helpers
from onnxflow.version import __version__ as onnxflow_version


UnionValidModelInstanceTypes = Union[
    None,
    str,
    torch.nn.Module,
    torch.jit.ScriptModule,
    "tf.keras.Model",
    sklearn.base.BaseEstimator,
]

if os.environ.get("MLAGILITY_ONNX_OPSET"):
    DEFAULT_ONNX_OPSET = int(os.environ.get("MLAGILITY_ONNX_OPSET"))
else:
    DEFAULT_ONNX_OPSET = 17

MINIMUM_ONNX_OPSET = 13

DEFAULT_CACHE_DIR = os.getcwd()
DEFAULT_REBUILD_POLICY = "if_needed"


class Backend(enum.Enum):
    AUTO = "auto"
    LOCAL = "local"
    CLOUD = "cloud"
    REMOTE = "remote"


class ModelType(enum.Enum):
    PYTORCH = "pytorch"
    PYTORCH_COMPILED = "pytorch_compiled"
    KERAS = "keras"
    ONNX_FILE = "onnx_file"
    HUMMINGBIRD = "hummingbird"
    UNKNOWN = "unknown"


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf8") as stream:
        try:
            return yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            raise exp.IOError(
                f"Failed while trying to open {file_path}."
                f"The exception that triggered this was:\n{e}"
            )


def output_dir(cache_dir, build_name):
    path = os.path.join(cache_dir, build_name)
    return path


def state_file(cache_dir, build_name):
    state_file_name = f"{build_name}_state.yaml"
    path = os.path.join(output_dir(cache_dir, build_name), state_file_name)
    return path


def hash_model(model, model_type: ModelType, hash_params: bool = True):
    # If the model is a path to a file, hash the file
    if model_type == ModelType.ONNX_FILE:
        # TODO: Implement a way of hashing the models but not the parameters
        # of ONNX inputs.
        if not hash_params:
            msg = "hash_params must be True for model_type ONNX_FILE"
            raise ValueError(msg)
        if os.path.isfile(model):
            with open(model, "rb") as f:
                file_content = f.read()
            return hashlib.sha256(file_content).hexdigest()
        else:
            raise ValueError(
                "hash_model received str model that doesn't correspond to a file"
            )

    elif model_type in [ModelType.PYTORCH, ModelType.PYTORCH_COMPILED]:
        # Convert model parameters and topology to string
        hashable_params = {}
        for name, param in model.named_parameters():
            hashable_params[name] = param.data
        if hash_params:
            hashable_model = (str(model) + str(hashable_params)).encode()
        else:
            hashable_model = str(model).encode()

        # Return hash of topology and parameters
        return hashlib.sha256(hashable_model).hexdigest()

    elif model_type == ModelType.KERAS:
        # Convert model parameters and topology to string
        summary_list = []  # type: List[str]

        # pylint: disable=unnecessary-lambda
        model.summary(print_fn=lambda x: summary_list.append(x))

        summary_str = " ".join(summary_list)
        hashable_params = {}
        for layer in model.layers:
            hashable_params[layer.name] = layer.weights
        if hash_params:
            hashable_model = (summary_str + str(hashable_params)).encode()
        else:
            hashable_model = summary_str.encode()

        # Return hash of topology and parameters
        return hashlib.sha256(hashable_model).hexdigest()

    elif model_type == ModelType.HUMMINGBIRD:
        import pickle

        return hashlib.sha256(pickle.dumps(model)).hexdigest()

    else:
        msg = f"""
        model_type "{model_type}" unsupported by this hash_model function
        """
        raise ValueError(msg)


class Status(enum.Enum):
    NOT_STARTED = "not_started"
    PARTIAL_BUILD = "partial_build"
    BUILD_RUNNING = "build_running"
    SUCCESSFUL_BUILD = "successful_build"
    FAILED_BUILD = "failed_build"


# Create a unique ID from this run by hashing pid + process start time
def unique_id():
    pid = os.getpid()
    p = psutil.Process(pid)
    start_time = p.create_time()
    return hashlib.sha256(f"{pid}{start_time}".encode()).hexdigest()


def get_shapes_and_dtypes(inputs: dict):
    """
    Return the shape and data type of each value in the inputs dict
    """
    shapes = {}
    dtypes = {}
    for key in sorted(inputs):
        value = inputs[key]
        if isinstance(
            value,
            (list, tuple),
        ):
            for v, i in zip(value, range(len(value))):
                subkey = f"{key}[{i}]"
                shapes[subkey] = np.array(v).shape
                dtypes[subkey] = np.array(v).dtype.name
        elif torch.is_tensor(value):
            shapes[key] = np.array(value.detach()).shape
            dtypes[key] = np.array(value.detach()).dtype.name
        elif tf_helpers.is_keras_tensor(value):
            shapes[key] = np.array(value).shape
            dtypes[key] = np.array(value).dtype.name
        elif isinstance(value, np.ndarray):
            shapes[key] = value.shape
            dtypes[key] = value.dtype.name
        elif isinstance(value, (bool, int, float)):
            shapes[key] = (1,)
            dtypes[key] = type(value).__name__
        elif value is None:
            pass
        else:
            raise exp.Error(
                "One of the provided inputs contains the unsupported "
                f' type {type(value)} at key "{key}".'
            )

    return shapes, dtypes


@dataclasses.dataclass(frozen=True)
class Config:
    """
    User-provided build configuration. Instances of Config should not be modified
    once they have been instantiated (frozen=True enforces this).

    Note: modifying this struct can create a breaking change that
    requires users to rebuild their models. Increment the minor
    version number of the onnxflow package if you do make a build-
    breaking change.
    """

    build_name: str
    auto_name: bool
    sequence: List[str]
    onnx_opset: int


@dataclasses.dataclass
class Info:
    """
    Information about a build that may be useful for analysis
    or debugging purposes.

    Note: There is no guarantee that members of this class will
    have non-None values at the end of a build. Do
    not take a dependence on any member of this class.
    """

    backend: Backend = Backend.AUTO
    base_onnx_exported: Optional[bool] = None
    opt_onnx_exported: Optional[bool] = None
    opt_onnx_ops: Optional[List[str]] = None
    converted_onnx_exported: Optional[bool] = None
    quantized_onnx_exported: Optional[bool] = None
    skipped_stages: int = 0
    all_build_stages: List[str] = dataclasses.field(default_factory=list)
    current_build_stage: str = None
    completed_build_stages: List[str] = dataclasses.field(default_factory=list)
    build_stage_execution_times: Dict[str, float] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class State:
    # User-provided args that influence the generated model
    config: Config

    # User-provided args that do not influence the generated model
    monitor: bool = False
    rebuild: str = ""
    cache_dir: str = ""

    # User-provided args that will not be saved as part of state.yaml
    model: UnionValidModelInstanceTypes = None
    inputs: Optional[Dict[str, Any]] = None

    # Optional information about the build
    info: Info = Info()

    # Member variable that helps the code know if State has called
    # __post_init__ yet
    save_when_setting_attribute: bool = False

    # All of the following are critical aspects of the build,
    # including properties of the tool and choices made
    # while building the model, which determine the outcome of the build.
    # NOTE: adding or changing a member name in this struct can create
    # a breaking change that requires users to rebuild their models.
    # Increment the minor version number of the onnxflow package if you
    # do make a build-breaking change.

    onnxflow_version: str = onnxflow_version
    model_type: ModelType = ModelType.UNKNOWN
    uid: Optional[int] = None
    model_hash: Optional[int] = None
    build_status: Status = Status.NOT_STARTED
    expected_input_shapes: Optional[Dict[str, list]] = None
    expected_input_dtypes: Optional[Dict[str, list]] = None
    expected_output_names: Optional[List] = None

    # Whether or not inputs must be downcasted during inference
    downcast_applied: bool = False

    # The results of the most recent stage that was executed
    intermediate_results: Any = None

    quantization_samples: Optional[Collection] = None

    def __post_init__(self):
        if self.uid is None:
            self.uid = unique_id()
        if self.inputs is not None:
            (
                self.expected_input_shapes,
                self.expected_input_dtypes,
            ) = get_shapes_and_dtypes(self.inputs)
        if self.model is not None and self.model_type != ModelType.UNKNOWN:
            self.model_hash = hash_model(self.model, self.model_type)

        self.save_when_setting_attribute = True

    def __setattr__(self, name, val):
        super().__setattr__(name, val)

        # Always automatically save the state.yaml whenever State is modified
        # But don't bother saving until after __post_init__ is done (indicated
        # by the save_when_setting_attribute flag)
        # Note: This only works when elements of the state are set directly.
        # When an element of state.info gets set, for example, state needs
        # to be explicitly saved by calling state.save().
        if self.save_when_setting_attribute and name != "save_when_setting_attribute":
            self.save()

    @property
    def original_inputs_file(self):
        return os.path.join(
            output_dir(self.cache_dir, self.config.build_name), "inputs.npy"
        )

    @property
    def onnx_dir(self):
        return os.path.join(output_dir(self.cache_dir, self.config.build_name), "onnx")

    @property
    def base_onnx_file(self):
        return os.path.join(
            self.onnx_dir,
            f"{self.config.build_name}-op{self.config.onnx_opset}-base.onnx",
        )

    @property
    def opt_onnx_file(self):
        return os.path.join(
            self.onnx_dir,
            f"{self.config.build_name}-op{self.config.onnx_opset}-opt.onnx",
        )

    @property
    def converted_onnx_file(self):
        return os.path.join(
            self.onnx_dir,
            f"{self.config.build_name}-op{self.config.onnx_opset}-opt-f16.onnx",
        )

    @property
    def quantized_onnx_file(self):
        return os.path.join(
            self.onnx_dir,
            f"{self.config.build_name}-op{self.config.onnx_opset}-opt-quantized_int8.onnx",
        )

    def prepare_file_system(self):
        # Create output folder if it doesn't exist
        os.makedirs(output_dir(self.cache_dir, self.config.build_name), exist_ok=True)
        os.makedirs(self.onnx_dir, exist_ok=True)

    def prepare_state_dict(self) -> Dict:
        state_dict = {
            key: value
            for key, value in vars(self).items()
            if not key == "inputs"
            and not key == "model"
            and not key == "save_when_setting_attribute"
        }

        # Special case for saving objects
        state_dict["config"] = copy.deepcopy(vars(self.config))
        state_dict["info"] = copy.deepcopy(vars(self.info))

        state_dict["model_type"] = self.model_type.value
        state_dict["build_status"] = self.build_status.value

        state_dict["info"]["backend"] = self.info.backend.value

        # During actual execution, quantization_samples in the state
        # stores the actual quantization samples.
        # However, we do not save quantization samples
        # Instead, we save a boolean to indicate whether the model
        # stored has been quantized by some samples.
        if self.quantization_samples:
            state_dict["quantization_samples"] = True
        else:
            state_dict["quantization_samples"] = False

        return state_dict

    def save_yaml(self, state_dict: Dict):
        with open(
            state_file(self.cache_dir, self.config.build_name), "w", encoding="utf8"
        ) as outfile:
            yaml.dump(state_dict, outfile)

    def save(self):
        self.prepare_file_system()

        state_dict = self.prepare_state_dict()

        self.save_yaml(state_dict)


def load_state(
    cache_dir=DEFAULT_CACHE_DIR,
    build_name=None,
    state_path=None,
    state_type: Type = State,
) -> State:
    if state_path is not None:
        file_path = state_path
    elif build_name is not None:
        file_path = state_file(cache_dir, build_name)
    else:
        raise ValueError(
            "Only build_name or state_path should be set, not both or neither"
        )

    state_dict = load_yaml(file_path)

    # Get the type of Config and Info in case they have been overloaded
    field_types = {field.name: field.type for field in dataclasses.fields(state_type)}
    config_type = field_types["config"]
    info_type = field_types["info"]

    try:
        # Special case for loading enums
        state_dict["model_type"] = ModelType(state_dict["model_type"])
        state_dict["build_status"] = Status(state_dict["build_status"])
        state_dict["config"] = config_type(**state_dict["config"])

        # The info section is meant to be forwards compatible with future
        # version of onnxflow. Fields available in the state.yaml are copied
        # in to the new State instance, and all other fields are left
        # to their default value. Fields that existed in a previous version
        # of onnxflow, but have since been removed, are ignored.

        info_tmp = {}
        for key, value in state_dict["info"].items():
            info_keys = [field.name for field in dataclasses.fields(info_type)]
            if key in info_keys:
                if key == "backend":
                    info_tmp["backend"] = Backend(value)
                else:
                    info_tmp[key] = value

        state_dict["info"] = info_type(**info_tmp)

        state = state_type(**state_dict)

    except (KeyError, TypeError) as e:
        if state_path is not None:
            path_suggestion = pathlib.Path(state_path).parent
        else:
            path_suggestion = output_dir(cache_dir, build_name)
        msg = f"""
        The cached build of this model was built with an
        incompatible older version of the tool.

        Suggested solution: delete the build with
        rm -rf {path_suggestion}

        The underlying code raised this exception:
        {e}
        """
        raise exp.StateError(msg)

    return state


class Logger:
    """
    Redirects stdout to to file (and console if needed)
    """

    def __init__(self, log_path=None):
        self.debug = os.environ.get("ONNXFLOW_DEBUG") == "True"
        self.terminal = sys.stdout
        self.log_file = (
            None if log_path is None else open(log_path, "w", encoding="utf8")
        )

    def write(self, message):
        if self.log_file is not None:
            self.log_file.write(message)
        if self.debug or self.log_file is None:
            self.terminal.write(message)
            self.terminal.flush()

    def flush(self):
        # needed for python 3 compatibility.
        pass
