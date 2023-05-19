# labels: test_group::monthly author::huggingface name::xlm-roberta-large downloads::1,014,718 license::mit task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='xlm-roberta-large')
unmasker("Hello I'm a <mask> model.")


