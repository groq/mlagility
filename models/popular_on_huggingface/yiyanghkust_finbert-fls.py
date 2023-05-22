# labels: test_group::monthly author::yiyanghkust name::finbert-fls downloads::2,163 task::Natural_Language_Processing sub_task::Text_Classification
# tested in transformers==4.18.0 
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-fls',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-fls')
nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
results = nlp('We expect the age of our fleet to enhance availability and reliability due to reduced downtime for repairs.')
print(results)  # [{'label': 'Specific FLS', 'score': 0.77278733253479}]
