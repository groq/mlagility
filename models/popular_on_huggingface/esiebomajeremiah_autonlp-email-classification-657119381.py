# labels: test_group::monthly author::esiebomajeremiah name::autonlp-email-classification-657119381 downloads::1,890 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("esiebomajeremiah/autonlp-email-classification-657119381", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("esiebomajeremiah/autonlp-email-classification-657119381", use_auth_token=True)

inputs = tokenizer("I love AutoNLP", return_tensors="pt")

outputs = model(**inputs)
