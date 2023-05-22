# labels: test_group::monthly author::j-hartmann name::emotion-english-distilroberta-base downloads::566,562 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
classifier("I love this!")
