# labels: test_group::monthly author::huggingface name::bert-large-cased downloads::421,447 license::apache-2.0 task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-large-cased')
unmasker("Hello I'm a [MASK] model.")

