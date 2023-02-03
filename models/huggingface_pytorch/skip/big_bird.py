# labels: name::big_bird author::huggingface_pytorch
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size = parse(["batch_size"])


# Model and input configurations
config = transformers.BigBirdConfig(attention_type="block_sparse")
model = transformers.BigBirdModel(config)
inputs = {
    "input_ids": torch.ones(batch_size, 1024, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, 1024, dtype=torch.float),
}


# Call model
model(**inputs)
