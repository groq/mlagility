# labels: test_group::monthly author::j-hartmann name::purchase-intention-english-roberta-large downloads::181 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
classifier = pipeline("text-classification", model="j-hartmann/purchase-intention-english-roberta-large", return_all_scores=True)
classifier("I want this!")
