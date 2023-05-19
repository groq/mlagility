# labels: test_group::monthly author::CAMeL-Lab name::bert-base-arabic-camelbert-mix-pos-msa downloads::2,917 license::apache-2.0 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import pipeline
pos = pipeline('token-classification', model='CAMeL-Lab/bert-base-arabic-camelbert-mix-pos-msa')
text = 'إمارة أبوظبي هي إحدى إمارات دولة الإمارات العربية المتحدة السبع'
pos(text)

