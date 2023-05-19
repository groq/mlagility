# labels: test_group::monthly author::siebert name::sentiment-roberta-large-english downloads::172,559 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
print(sentiment_analysis("I love this!"))
