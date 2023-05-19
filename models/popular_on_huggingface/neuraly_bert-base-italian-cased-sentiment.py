# labels: test_group::monthly author::neuraly name::bert-base-italian-cased-sentiment downloads::1,736 license::mit task::Natural_Language_Processing sub_task::Text_Classification
import torch
from torch import nn  
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("neuraly/bert-base-italian-cased-sentiment")
# Load the model, use .cuda() to load it on the GPU
model = AutoModelForSequenceClassification.from_pretrained("neuraly/bert-base-italian-cased-sentiment")

sentence = 'Huggingface è un team fantastico!'
input_ids = tokenizer.encode(sentence, add_special_tokens=True)

# Create tensor, use .cuda() to transfer the tensor to GPU
tensor = torch.tensor(input_ids).long()
# Fake batch dimension
tensor = tensor.unsqueeze(0)

# Call the model and get the logits
logits, = model(tensor)

# Remove the fake batch dimension
logits = logits.squeeze(0)

# The model was trained with a Log Likelyhood + Softmax combined loss, hence to extract probabilities we need a softmax on top of the logits tensor
proba = nn.functional.softmax(logits, dim=0)

# Unpack the tensor to obtain negative, neutral and positive probabilities
negative, neutral, positive = proba
