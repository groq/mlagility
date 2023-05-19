# labels: test_group::mlagility name::retribert author::huggingface_pytorch task::Natural_Language_Processing
from mlagility.parser import parse
import transformers
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.RetriBertConfig()
model = transformers.RetriBertModel(config)
inputs = {
    "input_ids_query": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "input_ids_doc": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask_query": torch.ones(batch_size, max_seq_length, dtype=torch.float),
    "attention_mask_doc": torch.ones(batch_size, max_seq_length, dtype=torch.float),
    "checkpoint_batch_size": -1,
}


# Call model
model(**inputs)
