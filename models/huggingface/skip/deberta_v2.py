# labels: name::deberta_v2 author::skip
from mlagility.parser import parse
import transformers
import torch

# Reason for skipping: estimated to require ~18 GroqChip1 processors (877M parameters),
#   and we skip any model that requires > 16 GroqChip1 processors

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.DebertaV2Config()
model = transformers.DebertaV2Model(config)
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
