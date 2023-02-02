# labels: name::swin author::skip
import mlagility
import transformers
import torch

# Parsing command-line arguments
batch_size, height, num_channels, width = mlagility.parse(
    ["batch_size", "height", "num_channels", "width"]
)


# Model and input configurations
config = transformers.SwinConfig()
model = transformers.SwinModel(config)
inputs = {
    "pixel_values": torch.ones(
        batch_size, num_channels, height, width, dtype=torch.float
    ),
}


# Call model
model(**inputs)
