# labels: test_group::mlagility name::layoutlm author::transformers task::Natural_Language_Processing
from mlagility.parser import parse
import transformers
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.LayoutLMConfig()
model = transformers.LayoutLMModel(config)
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
    "bbox": torch.zeros(
        tuple(list(torch.Size([batch_size, max_seq_length])) + [4]),
        dtype=torch.long,
    ),
}


# Call model
model(**inputs)
