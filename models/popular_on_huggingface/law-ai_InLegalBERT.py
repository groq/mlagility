# labels: test_group::monthly author::law-ai name::InLegalBERT downloads::382 license::mit task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
text = "Replace this string with yours"
encoded_input = tokenizer(text, return_tensors="pt")
model = AutoModel.from_pretrained("law-ai/InLegalBERT")
output = model(**encoded_input)
last_hidden_state = output.last_hidden_state
