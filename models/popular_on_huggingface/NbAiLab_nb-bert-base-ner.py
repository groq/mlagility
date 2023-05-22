# labels: test_group::monthly author::NbAiLab name::nb-bert-base-ner downloads::188 license::cc-by-4.0 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-base-ner")
model = AutoModelForTokenClassification.from_pretrained("NbAiLab/nb-bert-base-ner")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Jeg heter Kjell og bor i Oslo."

ner_results = nlp(example)
print(ner_results)
