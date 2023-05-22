# labels: test_group::monthly author::huggingface name::xlm-roberta-large-finetuned-conll03-german downloads::9,448 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-german")
model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-german")
classifier = pipeline("ner", model=model, tokenizer=tokenizer)
classifier("Bayern München ist wieder alleiniger Top-Favorit auf den Gewinn der deutschen Fußball-Meisterschaft.")


