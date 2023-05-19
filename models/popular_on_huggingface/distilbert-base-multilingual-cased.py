# labels: test_group::monthly author::huggingface name::distilbert-base-multilingual-cased downloads::1,001,625 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='distilbert-base-multilingual-cased')
unmasker("Hello I'm a [MASK] model.")


