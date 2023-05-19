# labels: test_group::monthly author::huggingface name::albert-xxlarge-v1 downloads::6,343 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='albert-xxlarge-v1')
unmasker("Hello I'm a [MASK] model.")

