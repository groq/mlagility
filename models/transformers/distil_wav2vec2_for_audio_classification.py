# labels: test_group::mlagility name::distil_wav2vec2_for_audio_classification author::transformers task::Audio
"""https://huggingface.co/bookbot/distil-wav2vec2-xls-r-adult-child-cls-89m"""
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, max_audio_seq_length = parse(
    ["batch_size", "max_audio_seq_length"]
)


# This version of distil-wav2vec2 performs audio classification

# Model and input configurations
model = transformers.AutoModelForAudioClassification.from_pretrained(
    "bookbot/distil-wav2vec2-adult-child-cls-37m"
)
inputs = {"input_values": torch.ones(batch_size, max_audio_seq_length)}


# Call model
model(**inputs)
