# labels: test_group::monthly author::uer name::roberta-base-word-chinese-cluecorpussmall downloads::485 task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='uer/roberta-medium-word-chinese-cluecorpussmall')
unmasker("[MASK]的首都是北京。")

