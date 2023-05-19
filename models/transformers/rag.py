# labels: test_group::mlagility name::rag author::huggingface_pytorch task::MultiModal
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
model = transformers.RagModel.from_pretrained("facebook/rag-token-base")
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "context_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
    "context_attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
    "doc_scores": torch.ones(batch_size, 5, dtype=torch.float),
}


# Call model
model(**inputs)
