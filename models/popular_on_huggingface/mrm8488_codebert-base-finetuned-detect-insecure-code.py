# labels: test_group::monthly author::mrm8488 name::codebert-base-finetuned-detect-insecure-code downloads::4,786 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
tokenizer = AutoTokenizer.from_pretrained('mrm8488/codebert-base-finetuned-detect-insecure-code')
model = AutoModelForSequenceClassification.from_pretrained('mrm8488/codebert-base-finetuned-detect-insecure-code')

inputs = tokenizer("your code here", return_tensors="pt", truncation=True, padding='max_length')
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

print(np.argmax(logits.detach().numpy()))
