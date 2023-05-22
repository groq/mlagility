# labels: test_group::mlagility name::mobilevit_small_for_semantic_segmentation author::transformers task::Computer_Vision
"""https://huggingface.co/apple/mobilevit-small"""
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, height, num_channels, width = parse(
    ["batch_size", "height", "num_channels", "width"]
)


# Model and input configurations
model = transformers.MobileViTForSemanticSegmentation.from_pretrained(
    "apple/deeplabv3-mobilevit-small"
)

inputs = {
    "pixel_values": torch.ones(
        batch_size, num_channels, height, width, dtype=torch.float
    ),
}


# Call model
model(**inputs)
