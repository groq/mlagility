# labels: name::albert author::skip
import mlagility
import transformers
import torch

# Reason for skipping: estimated to require ~49 GroqChip1 processors (2437M parameters),
#   and we skip any model that requires > 16 GroqChip1 processors

# Parsing command-line arguments
batch_size, max_seq_length = mlagility.parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.AlbertConfig()
model = transformers.AlbertModel(config)
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
