# labels: test_group::mlagility name::distilhubert_for_audio_classification author::transformers task::Audio
"""https://huggingface.co/anton-l/distilhubert-ft-common-language"""
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, max_audio_seq_length = parse(
    ["batch_size", "max_audio_seq_length"]
)


# This version of distil-hubert performs audio classification

# Model and input configurations
model = transformers.AutoModelForAudioClassification.from_pretrained(
    "anton-l/distilhubert-ft-common-language"
)
inputs = {"input_values": torch.ones(batch_size, max_audio_seq_length)}


# Call model
model(**inputs)
