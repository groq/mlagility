# labels: test_group::monthly author::Elron name::bleurt-tiny-512 downloads::34,866 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-tiny-512")
model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-tiny-512")
model.eval()

references = ["hello world", "hello world"]
candidates = ["hi universe", "bye world"]

with torch.no_grad():
  scores = model(**tokenizer(references, candidates, return_tensors='pt'))[0].squeeze()

print(scores) # tensor([-0.9414, -0.5678])
