from groqflow.version import __version__

from .api.api import benchit
from .analysis.analysis import main as autogroq
from .analysis.analysis import evaluate_script as evaluate
from .cli.cli import main as benchitcli
