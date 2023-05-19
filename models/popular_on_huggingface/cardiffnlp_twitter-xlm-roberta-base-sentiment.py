# labels: test_group::monthly author::cardiffnlp name::twitter-xlm-roberta-base-sentiment downloads::103,715 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
sentiment_task("T'estimo!")
