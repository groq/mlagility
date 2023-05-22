# labels: test_group::monthly author::VMware name::vbert-2021-base downloads::732 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('VMware/vbert-2021-base')
model = BertModel.from_pretrained("VMware/vbert-2021-base")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
