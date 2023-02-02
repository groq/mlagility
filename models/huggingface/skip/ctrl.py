# labels: name::ctrl author::skip
import mlagility
import transformers
import torch

# Parsing command-line arguments
batch_size, max_seq_length = mlagility.parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.CTRLConfig()
model = transformers.CTRLModel(config)
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
