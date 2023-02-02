# labels: name::vit_mae author::skip
import mlagility
import transformers
import torch

# Parsing command-line arguments
batch_size, height, num_channels, width = mlagility.parse(
    ["batch_size", "height", "num_channels", "width"]
)


# Model and input configurations
config = transformers.ViTMAEConfig()
model = transformers.ViTMAEModel(config)
inputs = {
    "pixel_values": torch.ones(
        batch_size, num_channels, height, width, dtype=torch.float
    ),
}


# Call model
model(**inputs)
