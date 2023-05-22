# labels: test_group::monthly author::huggingface name::xlm-mlm-en-2048 downloads::5,766 license::cc-by-nc-4.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import XLMTokenizer, XLMModel
import torch

tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-en-2048")
model = XLMModel.from_pretrained("xlm-mlm-en-2048")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
