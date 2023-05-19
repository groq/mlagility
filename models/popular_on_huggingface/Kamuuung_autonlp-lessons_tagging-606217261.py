# labels: test_group::monthly author::Kamuuung name::autonlp-lessons_tagging-606217261 downloads::292 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("Kamuuung/autonlp-lessons_tagging-606217261", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("Kamuuung/autonlp-lessons_tagging-606217261", use_auth_token=True)

inputs = tokenizer("I love AutoNLP", return_tensors="pt")

outputs = model(**inputs)
