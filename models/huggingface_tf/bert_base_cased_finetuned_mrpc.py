# labels: test_group::mlagility name::bert_base_cased_finetuned_mrpc author::huggingface_tf
"""
https://huggingface.co/bert-base-cased-finetuned-mrpc
"""

from mlagility.parser import parse
import transformers
import tensorflow as tf

tf.random.set_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


config = transformers.AutoConfig.from_pretrained("bert-base-cased-finetuned-mrpc")
model = transformers.TFAutoModel.from_config(config)
inputs = {
    "input_ids": tf.ones(shape=(batch_size, max_seq_length), dtype=tf.int32),
}

model(**inputs)


# Call model
model(**inputs)
