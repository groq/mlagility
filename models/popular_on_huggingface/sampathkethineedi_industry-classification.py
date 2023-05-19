# labels: test_group::monthly author::sampathkethineedi name::industry-classification downloads::7,659 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("sampathkethineedi/industry-classification")  
model = AutoModelForSequenceClassification.from_pretrained("sampathkethineedi/industry-classification")

industry_tags = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
industry_tags("Stellar Capital Services Limited is an India-based non-banking financial company ... loan against property, management consultancy, personal loans and unsecured loans.")

'''Ouput'''
[{'label': 'Consumer Finance', 'score': 0.9841355681419373}]
