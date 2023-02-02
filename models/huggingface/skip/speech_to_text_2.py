# labels: name::speech_to_text_2 author::skip
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.Speech2Text2Config(feature_size=80)
model = transformers.Speech2Text2PreTrainedModel(config)
inputs = {
    "input_features": torch.ones(batch_size, max_seq_length, 80, dtype=torch.float),
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
}


# Call model
model(**inputs)
