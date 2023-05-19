# labels: test_group::monthly author::Elron name::bleurt-base-512 downloads::243 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
model.eval()

references = ["hello world", "hello world"]
candidates = ["hi universe", "bye world"]

with torch.no_grad():
  scores = model(**tokenizer(references, candidates, return_tensors='pt'))[0].squeeze()

print(scores) # tensor([1.0327, 0.2055])
