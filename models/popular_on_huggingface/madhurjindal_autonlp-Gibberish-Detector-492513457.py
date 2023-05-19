# labels: test_group::monthly author::madhurjindal name::autonlp-Gibberish-Detector-492513457 downloads::11,891 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457", use_auth_token=True)

inputs = tokenizer("I love AutoNLP", return_tensors="pt")

outputs = model(**inputs)
