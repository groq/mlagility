# labels: test_group::monthly author::Jean-Baptiste name::roberta-large-ner-english downloads::142,964 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")


##### Process text sample (from wikipedia)

from transformers import pipeline

nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
nlp("Apple was founded in 1976 by Steve Jobs, Steve Wozniak and Ronald Wayne to develop and sell Wozniak's Apple I personal computer")


[{'entity_group': 'ORG',
  'score': 0.99381506,
  'word': ' Apple',
  'start': 0,
  'end': 5},
 {'entity_group': 'PER',
  'score': 0.99970853,
  'word': ' Steve Jobs',
  'start': 29,
  'end': 39},
 {'entity_group': 'PER',
  'score': 0.99981767,
  'word': ' Steve Wozniak',
  'start': 41,
  'end': 54},
 {'entity_group': 'PER',
  'score': 0.99956465,
  'word': ' Ronald Wayne',
  'start': 59,
  'end': 71},
 {'entity_group': 'PER',
  'score': 0.9997918,
  'word': ' Wozniak',
  'start': 92,
  'end': 99},
 {'entity_group': 'MISC',
  'score': 0.99956393,
  'word': ' Apple I',
  'start': 102,
  'end': 109}]
