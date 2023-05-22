# labels: test_group::monthly author::michiyasunaga name::LinkBERT-base downloads::372 license::apache-2.0 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/LinkBERT-base')
model = AutoModel.from_pretrained('michiyasunaga/LinkBERT-base')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
