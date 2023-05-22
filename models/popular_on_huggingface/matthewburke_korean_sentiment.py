# labels: test_group::monthly author::matthewburke name::korean_sentiment downloads::270 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
classifier = pipeline("text-classification", model="matthewburke/korean_sentiment")
custom_tweet = "영화 재밌다."
preds = classifier(custom_tweet, return_all_scores=True)
is_positive = preds[0][1]['score'] > 0.5
