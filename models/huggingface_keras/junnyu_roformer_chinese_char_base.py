# labels: test_group::mlagility name::junnyu_roformer_chinese_char_base author::huggingface_keras
"""
https://huggingface.co/junnyu/roformer_chinese_char_base
"""

from mlagility.parser import parse
import transformers
import tensorflow as tf

tf.random.set_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


config = transformers.AutoConfig.from_pretrained("junnyu/roformer_chinese_char_base")
model = transformers.TFAutoModel.from_config(config)
inputs = {
    "input_ids": tf.ones(shape=(batch_size, max_seq_length), dtype=tf.int32),
}

# Call model
model(**inputs)