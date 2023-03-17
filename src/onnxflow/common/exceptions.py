import onnxflow.common.printing as printing


class OnnxFlowError(Exception):
    """
    Indicates the user did something wrong with benchit()
    """

    def __init__(self, msg):
        super().__init__()
        printing.log_error(msg)


class BuilditCacheError(OnnxFlowError):
    """
    Indicates ambiguous behavior from
    when a build already exists in the OnnxFlow cache,
    but the model, inputs, or args have changed thereby invalidating
    the cached copy of the model.
    """


class BuilditEnvError(OnnxFlowError):
    """
    Indicates to the user that the required tools are not
    available on their PATH.
    """


class BuilditArgError(OnnxFlowError):
    """
    Indicates to the user that they provided invalid arguments to
    benchit()
    """


class BuilditStageError(Exception):
    """
    Let the user know that something went wrong while
    firing off a benchit() Stage.

    Note: not overloading __init__() so that the
    attempt to print to stdout isn't captured into
    the Stage's log file.
    """


class BuilditStateError(Exception):
    """
    Raised when something goes wrong with State
    """


class BuilditIntakeError(Exception):
    """
    Let the user know that something went wrong during the
    initial intake process of analyzing a model.
    """


class OnnxFlowIOError(OnnxFlowError):
    """
    Indicates to the user that an input/output operation failed,
    such trying to open a file.
    """


class ModelArgError(OnnxFlowError):
    """
    Indicates to the user that values provided to a Model instance method
    were not allowed.
    """


class ModelRuntimeError(OnnxFlowError):
    """
    Indicates to the user that attempting to invoke a Model instance failed.
    """
