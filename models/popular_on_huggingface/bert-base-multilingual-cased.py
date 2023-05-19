# labels: test_group::monthly author::huggingface name::bert-base-multilingual-cased downloads::2,263,070 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased')
unmasker("Hello I'm a [MASK] model.")


