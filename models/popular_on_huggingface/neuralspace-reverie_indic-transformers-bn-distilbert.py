# labels: test_group::monthly author::neuralspace-reverie name::indic-transformers-bn-distilbert downloads::3,064 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('neuralspace-reverie/indic-transformers-bn-distilbert')
model = AutoModel.from_pretrained('neuralspace-reverie/indic-transformers-bn-distilbert')
text = "আপনি কেমন আছেন?"
input_ids = tokenizer(text, return_tensors='pt')['input_ids']
out = model(input_ids)[0]
print(out.shape)
# out = [1, 5, 768] 
