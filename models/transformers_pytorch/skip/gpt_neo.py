# labels: name::gpt_neo author::huggingface_pytorch
from mlagility.parser import parse
import transformers
import torch

# Reason for skipping: estimated to require ~27 GroqChip1 processors (1315M parameters),
#   and we skip any model that requires > 16 GroqChip1 processors

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.GPTNeoConfig()
model = transformers.GPTNeoModel(config)
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
