# labels: test_group::mlagility name::marian author::huggingface
import mlagility
import transformers
import torch

# Parsing command-line arguments
batch_size, max_seq_length = mlagility.parse(["batch_size", "max_seq_length"])


# Model and input configurations
model = transformers.MarianModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
}


# Call model
model(**inputs)
