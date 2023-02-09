# labels: test_group::monthly author::huggingface name::roberta-large downloads::7,516,504 license::mit task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='roberta-large')
unmasker("Hello I'm a <mask> model.")


