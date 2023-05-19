# labels: test_group::mlagility name::mobilevit_x_small_for_semantic_segmentation author::transformers task::Computer_Vision
"""https://huggingface.co/apple/deeplabv3-mobilevit-x-small"""
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, height, num_channels, width = parse(
    ["batch_size", "height", "num_channels", "width"]
)


# Model and input configurations
model = transformers.MobileViTForSemanticSegmentation.from_pretrained(
    "apple/deeplabv3-mobilevit-x-small"
)

inputs = {
    "pixel_values": torch.ones(
        batch_size, num_channels, height, width, dtype=torch.float
    ),
}


# Call model
model(**inputs)
