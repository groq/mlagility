# labels: test_group::monthly author::huggingface name::bert-base-multilingual-uncased downloads::920,867 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-multilingual-uncased')
unmasker("Hello I'm a [MASK] model.")


