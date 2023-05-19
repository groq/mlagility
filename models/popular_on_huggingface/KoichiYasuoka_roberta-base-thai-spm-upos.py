# labels: test_group::monthly author::KoichiYasuoka name::roberta-base-thai-spm-upos downloads::728 license::apache-2.0 task::Natural_Language_Processing sub_task::Token_Classification
import torch
from transformers import AutoTokenizer,AutoModelForTokenClassification
tokenizer=AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-base-thai-spm-upos")
model=AutoModelForTokenClassification.from_pretrained("KoichiYasuoka/roberta-base-thai-spm-upos")
s="หลายหัวดีกว่าหัวเดียว"
t=tokenizer.tokenize(s)
p=[model.config.id2label[q] for q in torch.argmax(model(tokenizer.encode(s,return_tensors="pt"))["logits"],dim=2)[0].tolist()[1:-1]]
print(list(zip(t,p)))
