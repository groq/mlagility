# labels: test_group::monthly author::ml6team name::bert-base-uncased-city-country-ner downloads::36,981 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("ml6team/bert-base-uncased-city-country-ner")

model = AutoModelForTokenClassification.from_pretrained("ml6team/bert-base-uncased-city-country-ner")

from transformers import pipeline

nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
nlp("My name is Kermit and I live in London.")
