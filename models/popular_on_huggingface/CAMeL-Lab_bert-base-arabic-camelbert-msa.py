# labels: test_group::monthly author::CAMeL-Lab name::bert-base-arabic-camelbert-msa downloads::455 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='CAMeL-Lab/bert-base-arabic-camelbert-msa')
unmasker("الهدف من الحياة هو [MASK] .")

