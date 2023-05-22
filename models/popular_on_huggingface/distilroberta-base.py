# labels: test_group::monthly author::huggingface name::distilroberta-base downloads::1,915,687 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='distilroberta-base')
unmasker("The man worked as a <mask>.")
unmasker("The woman worked as a <mask>.")

