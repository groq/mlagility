from mlagility.version import __version__

from .api.script_api import benchmark_script
from .api.model_api import benchmark_workload
from .cli.cli import main as benchitcli
