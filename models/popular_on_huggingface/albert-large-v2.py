# labels: test_group::monthly author::huggingface name::albert-large-v2 downloads::1,089,768 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='albert-large-v2')
unmasker("Hello I'm a [MASK] model.")

