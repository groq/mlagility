# labels: test_group::monthly author::wanyu name::IteraTeR-ROBERTA-Intention-Classifier downloads::290 task::Natural_Language_Processing sub_task::Text_Classification
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier")
model = AutoModelForSequenceClassification.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier")

id2label = {0: "clarity", 1: "fluency", 2: "coherence", 3: "style", 4: "meaning-changed"}

before_text = 'I likes coffee.'
after_text = 'I like coffee.'
model_input = tokenizer(before_text, after_text, return_tensors='pt')
model_output = model(**model_input)
softmax_scores = torch.softmax(model_output.logits, dim=-1)
pred_id = torch.argmax(softmax_scores)
pred_label = id2label[pred_id.int()]
