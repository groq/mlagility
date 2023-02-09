# labels: test_group::monthly author::huggingface name::distilbert-base-uncased downloads::7,601,463 license::apache-2.0 task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
unmasker("Hello I'm a [MASK] model.")


