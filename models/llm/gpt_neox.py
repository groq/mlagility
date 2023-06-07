# labels: name::gpt_neox author::transformers task::Natural_Language_Processing
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.GPTNeoXConfig()
model = transformers.GPTNeoXModel(config)

# Make sure the user's sequence length fits within the model's maximum
assert max_seq_length <= config.max_position_embeddings


inputs = {
    "hidden_states": torch.ones(
        batch_size, max_seq_length, config.hidden_size, dtype=torch.float
    ),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
