# labels: test_group::monthly author::joeddav name::bart-large-mnli-yahoo-answers downloads::33,009 task::Natural_Language_Processing sub_task::Zero-Shot_Classification
from transformers import pipeline
nlp = pipeline("zero-shot-classification", model="joeddav/bart-large-mnli-yahoo-answers")

sequence_to_classify = "Who are you voting for in 2020?"
candidate_labels = ["Europe", "public health", "politics", "elections"]
hypothesis_template = "This text is about {}."
nlp(sequence_to_classify, candidate_labels, multi_class=True, hypothesis_template=hypothesis_template)
