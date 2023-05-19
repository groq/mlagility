# labels: test_group::monthly author::Davlan name::xlm-roberta-large-ner-hrl downloads::1,442 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("Davlan/xlm-roberta-large-ner-hrl")
model = AutoModelForTokenClassification.from_pretrained("Davlan/xlm-roberta-large-ner-hrl")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Nader Jokhadar had given Syria the lead with a well-struck header in the seventh minute."
ner_results = nlp(example)
print(ner_results)
