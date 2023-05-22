    # labels: test_group::monthly author::huggingface name::ctrl task::Natural_Language_Processing downloads::11,873 license::bsd-3-clause
from transformers import CTRLTokenizer, CTRLModel
import torch

tokenizer = CTRLTokenizer.from_pretrained("ctrl")
model = CTRLModel.from_pretrained("ctrl")

# CTRL was trained with control codes as the first token
inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)

