# labels: test_group::mlagility name::speech_to_text author::huggingface_pytorch task::Audio
from mlagility.parser import parse
import transformers
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.Speech2TextConfig(feature_size=80)
model = transformers.Speech2TextModel(config)
inputs = {
    "input_features": torch.ones(batch_size, max_seq_length, 80, dtype=torch.float),
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
}


# Call model
model(**inputs)
