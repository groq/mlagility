"""
Functions that help us avoid importing tensorflow (TF), since that import
takes a very long time.

The test `if "tensorflow" in sys.modules:` checks to see if TF
has already been imported. This will always be true if someone is passing
a TF model to groqflow, since... that TF model had to come from somewhere :)

If TF hasn't already been imported, then there is no change that an object
is a TF instance, or TF is in any particular mode, or anything else, so
we can just return False on those checks.
"""

import sys
from typing import List
import mlagility.analysis.util as util


def is_keras_subclass(obj_type) -> bool:
    if "tensorflow" in sys.modules:
        return issubclass(obj_type, sys.modules["tensorflow"].keras.Model)
    else:
        return False


def get_transformers_activations() -> List:
    """
    We need this helper because `import transformers.activations` brings in `tensorflow`
    We can apply this helper because there is 0 chance of encountering a
    transformers activation if user code has not already imported transformers
    """
    if "transformers" in sys.modules:
        return util.get_classes(sys.modules["transformers"].activations)
    else:
        return []
