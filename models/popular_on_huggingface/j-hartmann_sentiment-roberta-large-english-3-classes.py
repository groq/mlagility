# labels: test_group::monthly author::j-hartmann name::sentiment-roberta-large-english-3-classes downloads::7,816 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
classifier = pipeline("text-classification", model="j-hartmann/sentiment-roberta-large-english-3-classes", return_all_scores=True)
classifier("This is so nice!")
