# labels: test_group::monthly author::huggingface name::roberta-base downloads::11,510,142 license::mit task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='roberta-base')
unmasker("Hello I'm a <mask> model.")


