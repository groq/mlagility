# labels: test_group::mlagility name::minilmv2 author::huggingface
"""https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"""
import mlagility
import torch
from transformers import AutoModel

# Parsing command-line arguments
batch_size, max_seq_length = mlagility.parse(["batch_size", "max_seq_length"])


# This version of MiniLM generates token embeddings

# Model and input configurations
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
inputs = {
    "input_ids": torch.ones((2 * batch_size, max_seq_length), dtype=torch.long),
    "token_type_ids": torch.ones((2 * batch_size, max_seq_length), dtype=torch.long),
    "attention_mask": torch.ones((2 * batch_size, max_seq_length), dtype=torch.bool),
}


# Call model
model(**inputs)
