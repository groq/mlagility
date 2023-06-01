# labels: name::gptj author::transformers task::Natural_Language_Processing
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.GPTJConfig()
model = transformers.GPTJModel(config)

# Make sure the user's sequence length fits within the model's maximuim
assert max_seq_length <= config.n_positions

# GPT-J layers are stored in model.h. All layers are identical
# decoder layers of type GPTJBlock, so it is fine to just take
# model.h[0] to grab one of them
layer = model.h[0]


inputs = {
    "hidden_state": torch.ones(
        batch_size, max_seq_length, config.n_embd, dtype=torch.float
    ),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call layer
layer(**inputs)
