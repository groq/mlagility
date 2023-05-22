# labels: test_group::monthly author::assemblyai name::bert-large-uncased-sst2 downloads::820 task::Natural_Language_Processing sub_task::Text_Classification
import torch.nn.functional as F 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("assemblyai/bert-large-uncased-sst2") 
model = AutoModelForSequenceClassification.from_pretrained("assemblyai/bert-large-uncased-sst2")

tokenized_segments = tokenizer(["AssemblyAI is the best speech-to-text API for modern developers with performance being second to none!"], return_tensors="pt", padding=True, truncation=True)
tokenized_segments_input_ids, tokenized_segments_attention_mask = tokenized_segments.input_ids, tokenized_segments.attention_mask
model_predictions = F.softmax(model(input_ids=tokenized_segments_input_ids, attention_mask=tokenized_segments_attention_mask)['logits'], dim=1)

print("Positive probability: "+str(model_predictions[0][1].item()*100)+"%")
print("Negative probability: "+str(model_predictions[0][0].item()*100)+"%")
