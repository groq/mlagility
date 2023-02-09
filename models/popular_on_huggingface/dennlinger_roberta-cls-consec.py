# labels: test_group::monthly author::dennlinger name::roberta-cls-consec downloads::1,000 task::Text_Classification
from transformers import pipeline
pipe = pipeline("text-classification", model="dennlinger/roberta-cls-consec")

pipe("{First paragraph} [SEP] {Second paragraph}")
