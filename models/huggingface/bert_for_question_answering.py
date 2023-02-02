# labels: test_group::mlagility name::bert_for_question_answering author::huggingface
import mlagility
import transformers
import torch

# Parsing command-line arguments
batch_size, max_seq_length = mlagility.parse(["batch_size", "max_seq_length"])


# This version of BERT performs question answering,
# while the default model outputs raw hidden states.

# Model and input configurations
model = transformers.BertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
    "token_type_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
}


# Call model
model(**inputs)
