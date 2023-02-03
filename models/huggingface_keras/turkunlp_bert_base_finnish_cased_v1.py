# labels: test_group::mlagility name::turkunlp_bert_base_finnish_cased_v1 author::huggingface_keras
"""
https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1
"""

from mlagility.parser import parse
import transformers
import tensorflow as tf

tf.random.set_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


config = transformers.AutoConfig.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
model = transformers.TFAutoModel.from_config(config)
inputs = {
    "input_ids": tf.ones(shape=(batch_size, max_seq_length), dtype=tf.int32),
}

model(**inputs)


# Call model
model(**inputs)
