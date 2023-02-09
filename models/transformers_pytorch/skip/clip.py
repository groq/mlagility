# labels: name::clip author::huggingface_pytorch
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, height, num_channels, width = parse(
    ["batch_size", "height", "num_channels", "width"]
)


# Model and input configurations
config = transformers.CLIPConfig()
model = transformers.CLIPModel(config)
inputs = {
    "input_ids": torch.ones(batch_size, 77, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, 77, dtype=torch.float),
    "pixel_values": torch.ones(
        batch_size, num_channels, height, width, dtype=torch.float
    ),
}


# Call model
model(**inputs)
