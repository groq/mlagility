# labels: test_group::monthly author::CAMeL-Lab name::bert-base-arabic-camelbert-ca-pos-egy downloads::371 license::apache-2.0 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import pipeline
pos = pipeline('token-classification', model='CAMeL-Lab/bert-base-arabic-camelbert-ca-pos-egy')
text = 'عامل ايه ؟'
pos(text)

