# labels: test_group::monthly author::MoritzLaurer name::DeBERTa-v3-large-mnli-fever-anli-ling-wanli downloads::5,584 license::mit task::Natural_Language_Processing sub_task::Zero-Shot_Classification
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
sequence_to_classify = "Angela Merkel is a politician in Germany and leader of the CDU"
candidate_labels = ["politics", "economy", "entertainment", "environment"]
output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
print(output)
