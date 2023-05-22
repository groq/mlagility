# labels: test_group::monthly author::ahmedrachid name::FinancialBERT-Sentiment-Analysis downloads::3,689 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

model = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentences = ["Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007 representing 7.7 % of net sales.",  
             "Bids or offers include at least 1,000 shares and the value of the shares must correspond to at least EUR 4,000.", 
             "Raute reported a loss per share of EUR 0.86 for the first half of 2009 , against EPS of EUR 0.74 in the corresponding period of 2008.", 
             ]
results = nlp(sentences)
print(results)

[{'label': 'positive', 'score': 0.9998133778572083},
 {'label': 'neutral', 'score': 0.9997822642326355},
 {'label': 'negative', 'score': 0.9877365231513977}]
