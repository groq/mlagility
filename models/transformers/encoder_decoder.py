# labels: test_group::mlagility name::encoder_decoder author::transformers task::MultiModal
from mlagility.parser import parse
import transformers
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.EncoderDecoderConfig.from_encoder_decoder_configs(
    transformers.BertConfig(), transformers.BertConfig()
)
model = transformers.EncoderDecoderModel(config)
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
}


# Call model
model(**inputs)
