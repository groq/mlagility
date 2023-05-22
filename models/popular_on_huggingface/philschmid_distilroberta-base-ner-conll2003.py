# labels: test_group::monthly author::philschmid name::distilroberta-base-ner-conll2003 downloads::496 license::apache-2.0 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("philschmid/distilroberta-base-ner-conll2003")
model = AutoModelForTokenClassification.from_pretrained("philschmid/distilroberta-base-ner-conll2003")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
example = "My name is Philipp and live in Germany"

nlp(example)
