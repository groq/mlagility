# labels: test_group::mlagility name::sew_d author::huggingface_pytorch task::Natural_Language_Processing
from mlagility.parser import parse
import transformers
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size = parse(["batch_size"])


# Model and input configurations
config = transformers.SEWDConfig()
model = transformers.SEWDModel(config)
inputs = {
    "input_values": torch.ones(batch_size, 10000, dtype=torch.float),
}


# Call model
model(**inputs)
