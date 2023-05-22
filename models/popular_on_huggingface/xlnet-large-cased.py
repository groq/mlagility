# labels: test_group::monthly author::huggingface name::xlnet-large-cased downloads::11,519 license::mit task::Natural_Language_Processing sub_task::Text_Generation
from transformers import XLNetTokenizer, XLNetModel

tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
model = XLNetModel.from_pretrained('xlnet-large-cased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
