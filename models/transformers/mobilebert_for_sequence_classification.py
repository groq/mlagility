# labels: test_group::mlagility name::mobilebert_for_sequence_classification author::transformers task::Natural_Language_Processing
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# This version of MobileBERT performs sequence classification,
# while the default model outputs raw hidden states.

# Model and input configurations
model = transformers.MobileBertForSequenceClassification.from_pretrained(
    "lordtt13/emo-mobilebert"
)
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
