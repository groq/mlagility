# labels: test_group::monthly author::huggingface name::albert-base-v2 downloads::2,161,430 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='albert-base-v2')
unmasker("Hello I'm a [MASK] model.")

