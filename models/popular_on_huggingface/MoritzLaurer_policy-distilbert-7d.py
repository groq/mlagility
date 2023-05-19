# labels: test_group::monthly author::MoritzLaurer name::policy-distilbert-7d downloads::188 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "MoritzLaurer/policy-distilbert-7d"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "The new variant first detected in southern England in September is blamed for sharp rises in levels of positive tests in recent weeks in London, south-east England and the east of England"

input = tokenizer(text, truncation=True, return_tensors="pt")
output = model(input["input_ids"])
# the output corresponds to the following labels:
# 0: external relations, 1: freedom and democracy, 2: political system, 3: economy, 4: welfare and quality of life, 5: fabric of society, 6: social groups

# output to dictionary
prediction = torch.softmax(output["logits"][0], -1).tolist()
label_names = ["external relations", "freedom and democracy", "political system", "economy", "welfare and quality of life", "fabric of society", "social groups"]
prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
print(prediction)
#{'external relations': 0.0, 'freedom and democracy': 0.0, 'political system': 0.9, 'economy': 0.4, 
# 'welfare and quality of life': 98.3, 'fabric of society': 0.3, 'social groups': 0.0}
