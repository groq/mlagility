# labels: test_group::monthly author::Davlan name::bert-base-multilingual-cased-masakhaner downloads::396 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-masakhaner")
model = AutoModelForTokenClassification.from_pretrained("Davlan/bert-base-multilingual-cased-masakhaner")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Emir of Kano turban Zhang wey don spend 18 years for Nigeria"
ner_results = nlp(example)
print(ner_results)
