# labels: test_group::monthly author::huggingface name::transfo-xl-wt103 downloads::21,107 task::Natural_Language_Processing sub_task::Text_Generation
from transformers import TransfoXLTokenizer, TransfoXLModel
import torch

tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
model = TransfoXLModel.from_pretrained("transfo-xl-wt103")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
