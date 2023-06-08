# labels: name::gptj author::transformers task::Generative_AI
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.GPTJConfig()
model = transformers.GPTJModel(config)

# Make sure the user's sequence length fits within the model's maximum
assert max_seq_length <= config.n_positions

inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
