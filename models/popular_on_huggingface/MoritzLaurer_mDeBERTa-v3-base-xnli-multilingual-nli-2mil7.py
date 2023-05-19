# labels: test_group::monthly author::MoritzLaurer name::mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 downloads::2,386 license::mit task::Natural_Language_Processing sub_task::Zero-Shot_Classification
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
sequence_to_classify = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
candidate_labels = ["politics", "economy", "entertainment", "environment"]
output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
print(output)
