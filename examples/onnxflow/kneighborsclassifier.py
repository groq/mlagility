"""
The following example trains an KNeighborsClassifier against random data
then builds it into an ONNX file.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # pylint: disable=import-error
from onnxflow import build_model

batch_size = 320

# Generate random points in a 10-dimensional space with binary labels
np.random.seed(0)
x = np.random.rand(1000, 10).astype(np.float32)
y = np.random.randint(2, size=1000)

# Perform a test/train split of the (random) dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=batch_size, random_state=0
)

# Fit the model using standard sklearn patterns
kn_model = KNeighborsClassifier(
    n_neighbors=10
)
kn_model.fit(x_train, y_train)

# Build the model
omodel = build_model(
    kn_model,
    {"input_0": x_test},
    cache_dir="~/.cache/onnxflow_test_cache",
)

# Print build results
print(f"Build status: {omodel.state.build_status}")
