# labels: test_group::monthly author::cointegrated name::rubert-base-cased-nli-threeway downloads::2,865 task::Natural_Language_Processing sub_task::Zero-Shot_Classification
# !pip install transformers sentencepiece --quiet
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_checkpoint = 'cointegrated/rubert-base-cased-nli-threeway'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()

text1 = 'Сократ - человек, а все люди смертны.'
text2 = 'Сократ никогда не умрёт.'
with torch.inference_mode():
    out = model(**tokenizer(text1, text2, return_tensors='pt').to(model.device))
    proba = torch.softmax(out.logits, -1).cpu().numpy()[0]
print({v: proba[k] for k, v in model.config.id2label.items()})
# {'entailment': 0.009525929, 'contradiction': 0.9332064, 'neutral': 0.05726764} 
