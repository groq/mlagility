# labels: test_group::monthly author::savasy name::bert-base-turkish-sentiment-cased downloads::2,311 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
sa= pipeline("sentiment-analysis", tokenizer=tokenizer, model=model)

p = sa("bu telefon modelleri çok kaliteli , her parçası çok özel bence")
print(p)
# [{'label': 'LABEL_1', 'score': 0.9871089}]
print(p[0]['label'] == 'LABEL_1')
# True

p = sa("Film çok kötü ve çok sahteydi")
print(p)
# [{'label': 'LABEL_0', 'score': 0.9975505}]
print(p[0]['label'] == 'LABEL_1')
# False
