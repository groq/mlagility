# labels: test_group::mlagility name::facebook_vit_mae_base author::huggingface_keras
"""
https://huggingface.co/facebook/vit-mae-base
"""

from mlagility.parser import parse
import transformers
import tensorflow as tf

tf.random.set_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


config = transformers.AutoConfig.from_pretrained("facebook/vit-mae-base")
model = transformers.TFAutoModel.from_config(config)
inputs = {
    "input_ids": tf.ones(shape=(batch_size, max_seq_length), dtype=tf.int32),
}

# Call model
model(**inputs)