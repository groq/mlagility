# labels: test_group::monthly author::MoritzLaurer name::DeBERTa-v3-base-mnli-fever-docnli-ling-2c downloads::2,891 license::mit task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c")
sequence_to_classify = "Angela Merkel is a politician in Germany and leader of the CDU"
candidate_labels = ["politics", "economy", "entertainment", "environment"]
output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
print(output)
