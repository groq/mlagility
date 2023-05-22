# labels: test_group::monthly author::edumunozsala name::beto_sentiment_analysis_es downloads::204 license::apache-2.0 task::Natural_Language_Processing sub_task::Text_Classification
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("edumunozsala/beto_sentiment_analysis_es")
model = AutoModelForSequenceClassification.from_pretrained("edumunozsala/beto_sentiment_analysis_es")

text ="Se trata de una película interesante, con un solido argumento y un gran interpretación de su actor principal"

input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
outputs = model(input_ids)
output = outputs.logits.argmax(1)
