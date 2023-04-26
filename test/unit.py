"""
Miscellaneous unit tests
"""

import unittest
import os
import mlagility.common.filesystem as filesystem


class Testing(unittest.TestCase):
    def test_000_models_dir(self):
        """
        Make sure that filesystem.MODELS_DIR points to mlagility_install/models
        """

        # Make sure the path is valid
        assert os.path.isdir(filesystem.MODELS_DIR)

        # Make sure the readme and a couple of corpora are in the directory
        models = os.listdir(filesystem.MODELS_DIR)
        assert "selftest" in models
        assert "transformers" in models
        assert "readme.md" in models


if __name__ == "__main__":
    unittest.main()
