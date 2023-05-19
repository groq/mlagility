# labels: test_group::monthly author::mrm8488 name::bert-spanish-cased-finetuned-ner downloads::253,145 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import pipeline

nlp_ner = pipeline(
    "ner",
    model="mrm8488/bert-spanish-cased-finetuned-ner",
    tokenizer=(
        'mrm8488/bert-spanish-cased-finetuned-ner',  
        {"use_fast": False}
))

text = 'Mis amigos están pensando viajar a Londres este verano'

nlp_ner(text)

#Output: [{'entity': 'B-LOC', 'score': 0.9998720288276672, 'word': 'Londres'}]
