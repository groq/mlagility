# labels: test_group::mlagility name::bert_generation author::transformers task::Natural_Language_Processing
from mlagility.parser import parse
import transformers
import torch

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
model = transformers.EncoderDecoderModel(
    encoder=transformers.BertGenerationEncoder.from_pretrained(
        "bert-large-uncased", bos_token_id=101, eos_token_id=102
    ),
    decoder=transformers.BertGenerationDecoder.from_pretrained(
        "bert-large-uncased",
        add_cross_attention=True,
        is_decoder=True,
        bos_token_id=101,
        eos_token_id=102,
    ),
)
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
}


# Call model
model(**inputs)
