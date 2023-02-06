# labels: test_group::monthly author::dennlinger name::bert-wiki-paragraphs downloads::209 task::Text_Classification
from transformers import pipeline
pipe = pipeline("text-classification", model="dennlinger/bert-wiki-paragraphs")

pipe("{First paragraph} [SEP] {Second paragraph}")
