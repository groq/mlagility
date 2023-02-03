# labels: name::visual_bert author::huggingface_pytorch
from mlagility.parser import parse
import transformers
import torch


# Parsing command-line arguments
(
    batch_size,
    max_seq_length,
    visual_embedding_dim,
    visual_embedding_dim,
    visual_seq_length,
) = parse(
    [
        "batch_size",
        "max_seq_length",
        "visual_embedding_dim",
        "visual_embedding_dim",
        "visual_seq_length",
    ]
)

# Model and input configurations
config = transformers.VisualBertConfig()
model = transformers.VisualBertModel(config)
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
    "visual_embeds": torch.ones(
        batch_size, visual_seq_length, visual_embedding_dim, dtype=torch.float
    ),
    "visual_token_type_ids": torch.ones(
        batch_size, visual_seq_length, dtype=torch.long
    ),
    "visual_attention_mask": torch.ones(
        batch_size, visual_seq_length, dtype=torch.float
    ),
}


# Call model
model(**inputs)
