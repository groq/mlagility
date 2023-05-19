# labels: test_group::mlagility name::distilbert_for_question_answering author::transformers task::Natural_Language_Processing
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# This version of DistilBERT performs question answering,
# while the default model outputs raw hidden states.

# Model and input configurations
model = transformers.DistilBertForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased-distilled-squad"
)
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
