# labels: test_group::mlagility name::bert_large_cased_whole_word_masking_finetuned_squad author::huggingface_tf
"""
https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad
"""

from mlagility.parser import parse
import transformers
import tensorflow as tf

tf.random.set_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


config = transformers.AutoConfig.from_pretrained(
    "bert-large-cased-whole-word-masking-finetuned-squad"
)
model = transformers.TFAutoModel.from_config(config)
inputs = {
    "input_ids": tf.ones(shape=(batch_size, max_seq_length), dtype=tf.int32),
}

model(**inputs)


# Call model
model(**inputs)
