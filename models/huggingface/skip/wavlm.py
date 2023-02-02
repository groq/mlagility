# labels: name::wavlm author::huggingface
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
model = transformers.WavLMModel.from_pretrained(
    "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
)
inputs = {
    "input_values": torch.ones(batch_size, 10000, dtype=torch.float),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
