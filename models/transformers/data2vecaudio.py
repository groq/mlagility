# labels: test_group::mlagility name::data2vecaudio author::transformers task::Audio
from mlagility.parser import parse
import transformers
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size = parse(["batch_size"])


# Model and input configurations
config = transformers.Data2VecAudioConfig()
model = transformers.Data2VecAudioModel(config)
inputs = {
    "input_values": torch.ones(batch_size, 10000, dtype=torch.float),
    "attention_mask": torch.ones(batch_size, 10000, dtype=torch.int32),
}


# Call model
model(**inputs)
