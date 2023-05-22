# labels: test_group::monthly author::tuhailong name::SimCSE-bert-base task::Natural_Language_Processing downloads::214
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained("tuhailong/SimCSE-bert-base")
tokenizer = AutoTokenizer.from_pretrained("tuhailong/SimCSE-bert-base")
sentences_str_list = ["今天天气不错的","天气不错的"]
inputs = tokenizer(sentences_str_list,return_tensors="pt", padding='max_length', truncation=True, max_length=32)
outputs = model(**inputs)

