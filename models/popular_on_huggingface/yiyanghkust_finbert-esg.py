# labels: test_group::monthly author::yiyanghkust name::finbert-esg downloads::2,795 task::Natural_Language_Processing sub_task::Text_Classification
# tested in transformers==4.18.0 
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
results = nlp('Rhonda has been volunteering for several years for a variety of charitable community programs.')
print(results) # [{'label': 'Social', 'score': 0.9906041026115417}]
