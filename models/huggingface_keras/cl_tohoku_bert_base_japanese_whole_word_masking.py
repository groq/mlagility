# labels: test_group::mlagility name::cl_tohoku_bert_base_japanese_whole_word_masking author::huggingface_keras
"""
https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking
"""

from mlagility.parser import parse
import transformers
import tensorflow as tf

tf.random.set_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


config = transformers.AutoConfig.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
model = transformers.TFAutoModel.from_config(config)
inputs = {
    "input_ids": tf.ones(shape=(batch_size, max_seq_length), dtype=tf.int32),
}

model(**inputs)


# Call model
model(**inputs)
