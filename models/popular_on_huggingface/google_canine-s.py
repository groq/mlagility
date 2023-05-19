# labels: test_group::monthly,daily author::google name::canine-s downloads::10,734 license::apache-2.0 task::Multimodal sub_task::Feature_Extraction
from transformers import CanineTokenizer, CanineModel

model = CanineModel.from_pretrained('google/canine-s')
tokenizer = CanineTokenizer.from_pretrained('google/canine-s')

inputs = ["Life is like a box of chocolates.", "You never know what you gonna get."]
encoding = tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt")

outputs = model(**encoding) # forward pass
pooled_output = outputs.pooler_output
sequence_output = outputs.last_hidden_state
