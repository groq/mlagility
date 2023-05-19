# labels: test_group::monthly author::gilf name::french-postag-model downloads::737 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("gilf/french-postag-model")
model = AutoModelForTokenClassification.from_pretrained("gilf/french-postag-model")

from transformers import pipeline

nlp_token_class = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)

nlp_token_class('Face à un choc inédit, les mesures mises en place par le gouvernement ont permis une protection forte et efficace des ménages')
