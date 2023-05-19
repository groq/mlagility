# labels: test_group::monthly author::huggingface name::bert-large-uncased-whole-word-masking downloads::79,207 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-large-uncased-whole-word-masking')
unmasker("Hello I'm a [MASK] model.")

