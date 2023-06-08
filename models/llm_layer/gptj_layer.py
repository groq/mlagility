# labels: name::gptj_layer author::transformers task::Generative_AI
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.GPTJConfig()
model = transformers.models.gptj.modeling_gptj.GPTJBlock(config)

# Make sure the user's sequence length fits within the model's maximum
assert max_seq_length <= config.n_positions


inputs = {
    "hidden_states": torch.ones(
        batch_size, max_seq_length, config.n_embd, dtype=torch.float
    ),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call layer
model(**inputs)
