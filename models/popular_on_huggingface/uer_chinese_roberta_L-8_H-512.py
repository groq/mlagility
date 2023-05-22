# labels: test_group::monthly author::uer name::chinese_roberta_L-8_H-512 downloads::2,120 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='uer/chinese_roberta_L-8_H-512')
unmasker("中国的首都是[MASK]京。")

