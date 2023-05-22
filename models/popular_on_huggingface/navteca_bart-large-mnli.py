# labels: test_group::monthly author::navteca name::bart-large-mnli downloads::1,332 license::mit task::Natural_Language_Processing sub_task::Zero-Shot_Classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load model & tokenizer
bart_model = AutoModelForSequenceClassification.from_pretrained('navteca/bart-large-mnli')
bart_tokenizer = AutoTokenizer.from_pretrained('navteca/bart-large-mnli')

# Get predictions
nlp = pipeline('zero-shot-classification', model=bart_model, tokenizer=bart_tokenizer)

sequence = 'One day I will see the world.'
candidate_labels = ['cooking', 'dancing', 'travel']

result = nlp(sequence, candidate_labels, multi_label=True)

print(result)

#{
#  "sequence": "One day I will see the world.",
#  "labels": [
#    "travel",
#    "dancing",
#    "cooking"
#  ],
#  "scores": [
#    0.9941897988319397,
#    0.0060537424869835,
#    0.0020010927692056
#  ]
#}
