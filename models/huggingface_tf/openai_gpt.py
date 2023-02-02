# labels: test_group::mlagility name::openai_gpt author::huggingface_tf
"""
https://huggingface.co/openai-gpt
"""

from mlagility.parser import parse
import transformers
import tensorflow as tf

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


model = transformers.TFAutoModel.from_pretrained("openai-gpt")

inputs = {
    "input_ids": tf.ones(shape=(batch_size, max_seq_length), dtype=tf.int32),
}


# Call model
model(**inputs)
