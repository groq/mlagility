# labels: name::blenderbot author::huggingface
from mlagility.parser import parse
import transformers
import torch

# Reason for skipping: estimated to require ~54 GroqChip1 processors (2696M parameters),
#   and we skip any model that requires > 16 GroqChip1 processors

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.BlenderbotConfig()
model = transformers.BlenderbotModel(config)
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
}


# Call model
model(**inputs)
