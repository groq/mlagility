# labels: test_group::monthly author::huggingface name::roberta-base downloads::11,510,142 license::mit task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='roberta-base')
unmasker("Hello I'm a <mask> model.")


