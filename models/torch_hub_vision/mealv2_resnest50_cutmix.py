# labels: test_group::mlagility name::mealv2_resnest50_cutmix author::torch_hub_vision
"""
https://github.com/pytorch/hub/blob/master/pytorch_vision_meal_v2.md
"""

import mlagility
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, num_channels, width, height = mlagility.parse(
    ["batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = torch.hub.load(
    "szq0214/MEAL-V2",
    "meal_v2",
    "mealv2_resnest50_cutmix",
    weights=None,
)
model.eval()
inputs = {"x": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
