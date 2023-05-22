# labels: test_group::monthly author::mrm8488 name::bert-base-german-finetuned-ler downloads::351 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import pipeline

nlp_ler = pipeline(
    "ner",
    model="mrm8488/bert-base-german-finetuned-ler",
    tokenizer="mrm8488/bert-base-german-finetuned-ler"
)

text = "Your German legal text here"

nlp_ler(text)
