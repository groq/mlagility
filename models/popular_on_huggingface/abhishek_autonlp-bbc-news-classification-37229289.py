# labels: test_group::monthly author::abhishek name::autonlp-bbc-news-classification-37229289 downloads::524 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("abhishek/autonlp-bbc-news-classification-37229289", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("abhishek/autonlp-bbc-news-classification-37229289", use_auth_token=True)

inputs = tokenizer("I love AutoNLP", return_tensors="pt")

outputs = model(**inputs)
