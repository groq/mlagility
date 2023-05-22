# labels: test_group::monthly author::cointegrated name::rubert-tiny-toxicity downloads::5,614 task::Natural_Language_Processing sub_task::Text_Classification
# !pip install transformers sentencepiece --quiet
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()
    
def text2toxicity(text, aggregate=True):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba

print(text2toxicity('я люблю нигеров', True))
# 0.9350118728093193

print(text2toxicity('я люблю нигеров', False))
# [0.9715758  0.0180863  0.0045551  0.00189755 0.9331106 ]

print(text2toxicity(['я люблю нигеров', 'я люблю африканцев'], True))
# [0.93501186 0.04156357]

print(text2toxicity(['я люблю нигеров', 'я люблю африканцев'], False))
# [[9.7157580e-01 1.8086294e-02 4.5550885e-03 1.8975559e-03 9.3311059e-01]
#  [9.9979788e-01 1.9048342e-04 1.5297388e-04 1.7452303e-04 4.1369814e-02]]
