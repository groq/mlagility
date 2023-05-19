# labels: test_group::monthly author::philschmid name::distilroberta-base-ner-wikiann-conll2003-4-class downloads::213 license::apache-2.0 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("philschmid/distilroberta-base-ner-wikiann-conll2003-4-class")
model = AutoModelForTokenClassification.from_pretrained("philschmid/distilroberta-base-ner-wikiann-conll2003-4-class")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
example = "My name is Philipp and live in Germany"

nlp(example)
