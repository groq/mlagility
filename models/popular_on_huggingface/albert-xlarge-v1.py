# labels: test_group::monthly author::huggingface name::albert-xlarge-v1 downloads::322 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='albert-xlarge-v1')
unmasker("Hello I'm a [MASK] model.")

