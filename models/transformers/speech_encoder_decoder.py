# labels: test_group::mlagility name::speech_encoder_decoder author::huggingface_pytorch task::Audio
from mlagility.parser import parse
import transformers
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.SpeechEncoderDecoderConfig.from_encoder_decoder_configs(
    transformers.Wav2Vec2Config(), transformers.BertConfig()
)
model = transformers.SpeechEncoderDecoderModel(config)
inputs = {
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "input_values": torch.ones(batch_size, 10000, dtype=torch.float),
}


# Call model
model(**inputs)
