# labels: test_group::monthly author::huggingface name::bert-large-cased-whole-word-masking downloads::5,706 license::apache-2.0 task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-large-cased-whole-word-masking')
unmasker("Hello I'm a [MASK] model.")

