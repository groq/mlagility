# labels: test_group::mlagility name::wav2vec2 author::huggingface_pytorch task::Audio
from mlagility.parser import parse
import transformers
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size = parse(["batch_size"])


# Model and input configurations
config = transformers.Wav2Vec2Config()
model = transformers.Wav2Vec2Model(config)
inputs = {
    "input_values": torch.ones(batch_size, 10000, dtype=torch.float),
}


# Call model
model(**inputs)
