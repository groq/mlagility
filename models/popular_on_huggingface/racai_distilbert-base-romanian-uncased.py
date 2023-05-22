# labels: test_group::monthly author::racai name::distilbert-base-romanian-uncased task::Natural_Language_Processing downloads::389 license::mit
from transformers import AutoTokenizer, AutoModel

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("racai/distilbert-base-romanian-uncased")
model = AutoModel.from_pretrained("racai/distilbert-base-romanian-uncased")

# tokenize a test sentence
input_ids = tokenizer.encode("aceasta este o propoziție de test.", add_special_tokens=True, return_tensors="pt")

# run the tokens trough the model
outputs = model(input_ids)

print(outputs)
