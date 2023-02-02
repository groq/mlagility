# labels: name::mt5_xl author::skip
import mlagility
import transformers
import torch

# Reason for skipping: estimated to require ~65 GroqChip1 processors (3230M parameters),
#   and we skip any model that requires > 16 GroqChip1 processors

# Parsing command-line arguments
batch_size, max_seq_length = mlagility.parse(["batch_size", "max_seq_length"])


# Model and input configurations
model = transformers.MT5Model.from_pretrained("google/mt5-xl")
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
}


# Call model
model(**inputs)
