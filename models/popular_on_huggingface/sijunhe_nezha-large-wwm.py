# labels: test_group::monthly author::sijunhe name::nezha-large-wwm downloads::184 license::afl-3.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import BertTokenizer, NezhaModel
tokenizer = BertTokenizer.from_pretrained("sijunhe/nezha-large-wwm")
model = NezhaModel.from_pretrained("sijunhe/nezha-large-wwm")
text = "我爱北京天安门"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
