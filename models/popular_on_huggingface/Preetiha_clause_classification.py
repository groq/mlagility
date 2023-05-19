# labels: test_group::monthly author::Preetiha name::clause_classification downloads::264 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("Preetiha/autotrain-clause-classification-812025458", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("Preetiha/autotrain-clause-classification-812025458", use_auth_token=True)

inputs = tokenizer("I love AutoTrain", return_tensors="pt")

outputs = model(**inputs)
