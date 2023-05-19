# labels: test_group::monthly author::cardiffnlp name::tweet-topic-21-single downloads::853 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax

    
MODEL = f"cardiffnlp/tweet-topic-21-single"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
class_mapping = model.config.id2label

text = "Tesla stock is on the rise!"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

scores = output[0][0].detach().numpy()
scores = softmax(scores)

# TF
#model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
#class_mapping = model.config.id2label
#text = "Tesla stock is on the rise!"
#encoded_input = tokenizer(text, return_tensors='tf')
#output = model(**encoded_input)
#scores = output[0][0]
#scores = softmax(scores)


ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = class_mapping[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")
