# labels: test_group::mlagility name::perceiver author::transformers task::MultiModal
from mlagility.parser import parse
import transformers
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.PerceiverConfig(d_model=max_seq_length)
model = transformers.PerceiverModel(config)
inputs = {
    "inputs": torch.ones(1, batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
