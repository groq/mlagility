# labels: test_group::monthly author::huggingface name::xlm-roberta-base downloads::33,888,031 license::mit task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='xlm-roberta-base')
unmasker("Hello I'm a <mask> model.")


