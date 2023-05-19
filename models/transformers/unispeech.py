# labels: test_group::mlagility name::unispeech author::huggingface_pytorch task::Audio
from mlagility.parser import parse
import transformers
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.UniSpeechConfig()
model = transformers.UniSpeechModel(config)
inputs = {
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "input_values": torch.ones(batch_size, 10000, dtype=torch.float),
}


# Call model
model(**inputs)
