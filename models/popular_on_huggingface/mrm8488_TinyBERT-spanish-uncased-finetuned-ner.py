# labels: test_group::monthly author::mrm8488 name::TinyBERT-spanish-uncased-finetuned-ner downloads::1,131 task::Natural_Language_Processing sub_task::Token_Classification
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

id2label = {
    "0": "B-LOC",
    "1": "B-MISC",
    "2": "B-ORG",
    "3": "B-PER",
    "4": "I-LOC",
    "5": "I-MISC",
    "6": "I-ORG",
    "7": "I-PER",
    "8": "O"
}

tokenizer = AutoTokenizer.from_pretrained('mrm8488/TinyBERT-spanish-uncased-finetuned-ner')
model = AutoModelForTokenClassification.from_pretrained('mrm8488/TinyBERT-spanish-uncased-finetuned-ner')
text ="Mis amigos están pensando viajar a Londres este verano."
input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)

outputs = model(input_ids)
last_hidden_states = outputs[0]

for m in last_hidden_states:
  for index, n in enumerate(m):
    if(index > 0 and index <= len(text.split(" "))):
      print(text.split(" ")[index-1] + ": " + id2label[str(torch.argmax(n).item())])
      
'''
Output:
--------
Mis: O
amigos: O
están: O
pensando: O
viajar: O
a: O
Londres: B-LOC
este: O
verano.: O
'''
