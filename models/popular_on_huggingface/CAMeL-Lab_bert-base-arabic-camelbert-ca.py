# labels: test_group::monthly author::CAMeL-Lab name::bert-base-arabic-camelbert-ca downloads::1,406 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='CAMeL-Lab/bert-base-arabic-camelbert-ca')
unmasker("الهدف من الحياة هو [MASK] .")

