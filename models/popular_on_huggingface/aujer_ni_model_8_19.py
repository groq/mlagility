# labels: test_group::monthly author::aujer name::ni_model_8_19 downloads::236 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("aujer/autotrain-not_interested_8_19-1283149075", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("aujer/autotrain-not_interested_8_19-1283149075", use_auth_token=True)

inputs = tokenizer("I love AutoTrain", return_tensors="pt")

outputs = model(**inputs)
