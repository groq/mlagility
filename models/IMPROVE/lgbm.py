# labels: test_group::improve name::lgbm::regression
"""
https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor
"""
from mlagility.parser import parse
import lightgbm
import numpy
from sklearn.datasets import make_regression

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])

# scikit learn models use predict not call
class LightWrapper(lightgbm.LGBMRegressor):
    def __init__(self):
        super().__init__(n_features=50, boosting_type=None)
        
    def __call__(self, input):
        return self.predict(input)

model = LightWrapper()
x, y = make_regression(n_samples=1000, n_features=50)

# trying to hack together an inference without training
model.fitted_ = True
model._Booster = lightgbm.Booster(train_set = lightgbm.Dataset(x))

input=x[0].reshape(1,-1)

model(input)