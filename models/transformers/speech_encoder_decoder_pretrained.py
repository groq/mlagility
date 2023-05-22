# labels: test_group::mlagility name::speech_encoder_decoder_pretrained author::huggingface_pytorch task::Audio
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# This version of SpeechEncoderDecoderModel uses Speech2TextForConditionalGeneration
# as the decoder, while the other model uses BERT.

# Model and input configurations
model = transformers.SpeechEncoderDecoderModel.from_pretrained(
    "facebook/s2t-wav2vec2-large-en-de"
)
inputs = {
    "input_values": torch.ones(batch_size, max_seq_length, dtype=torch.float),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "return_dict": False,
    "output_attentions": False,
}


# Call model
model(**inputs)
