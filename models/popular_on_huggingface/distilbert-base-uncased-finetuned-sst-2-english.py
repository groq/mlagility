# labels: test_group::monthly author::huggingface name::distilbert-base-uncased-finetuned-sst-2-english downloads::2,161,565 license::apache-2.0 task::Natural_Language_Processing sub_task::Text_Classification
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]
