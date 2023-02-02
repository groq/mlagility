# labels: test_group::mlagility name::facebook_deit_base_distilled_patch16_224 author::huggingface_tf
"""
https://huggingface.co/facebook/deit-base-distilled-patch16-224
"""

from mlagility.parser import parse
import transformers
import tensorflow as tf

tf.random.set_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


config = transformers.AutoConfig.from_pretrained(
    "facebook/deit-base-distilled-patch16-224"
)
model = transformers.TFAutoModel.from_config(config)
inputs = {
    "input_ids": tf.ones(shape=(batch_size, max_seq_length), dtype=tf.int32),
}

model(**inputs)


# Call model
model(**inputs)
