# labels: test_group::mlagility name::wietsedv_bert_base_dutch_cased author::huggingface_tf
"""
https://huggingface.co/wietsedv/bert-base-dutch-cased
"""

import mlagility
import transformers
import tensorflow as tf

tf.random.set_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = mlagility.parse(["batch_size", "max_seq_length"])


config = transformers.AutoConfig.from_pretrained("wietsedv/bert-base-dutch-cased")
model = transformers.TFAutoModel.from_config(config)
inputs = {
    "input_ids": tf.ones(shape=(batch_size, max_seq_length), dtype=tf.int32),
}

model(**inputs)


# Call model
model(**inputs)
