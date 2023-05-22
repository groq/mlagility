# labels: test_group::monthly author::huggingface name::distilbert-base-cased task::Natural_Language_Processing downloads::393,397 license::apache-2.0
from transformers import pipeline
unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
unmasker("Hello I'm a [MASK] model.")


