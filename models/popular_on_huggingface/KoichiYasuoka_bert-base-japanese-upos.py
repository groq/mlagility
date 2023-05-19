# labels: test_group::monthly author::KoichiYasuoka name::bert-base-japanese-upos downloads::284 license::cc-by-sa-4.0 task::Natural_Language_Processing sub_task::Token_Classification
import torch
from transformers import AutoTokenizer,AutoModelForTokenClassification
tokenizer=AutoTokenizer.from_pretrained("KoichiYasuoka/bert-base-japanese-upos")
model=AutoModelForTokenClassification.from_pretrained("KoichiYasuoka/bert-base-japanese-upos")
s="国境の長いトンネルを抜けると雪国であった。"
p=[model.config.id2label[q] for q in torch.argmax(model(tokenizer.encode(s,return_tensors="pt"))["logits"],dim=2)[0].tolist()[1:-1]]
print(list(zip(s,p)))
