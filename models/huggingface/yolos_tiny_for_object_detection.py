# labels: test_group::mlagility name::yolos_tiny_for_object_detection author::huggingface
"""https://huggingface.co/hustvl/yolos-tiny"""
import mlagility
import transformers
import torch

# Parsing command-line arguments
batch_size, height, num_channels, width = mlagility.parse(
    ["batch_size", "height", "num_channels", "width"]
)


# Model and input configurations
model = transformers.YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")

inputs = {
    "pixel_values": torch.ones(
        [batch_size, num_channels, height, width], dtype=torch.float
    )
}


# Call model
model(**inputs)
