# labels: test_group::monthly author::daveni name::twitter-xlm-roberta-emotion-es downloads::664 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
model_path = "daveni/twitter-xlm-roberta-emotion-es"
emotion_analysis = pipeline("text-classification", framework="pt", model=model_path, tokenizer=model_path)
emotion_analysis("Einstein dijo: Solo hay dos cosas infinitas, el universo y los pinches anuncios de bitcoin en Twitter. Paren ya carajo aaaaaaghhgggghhh me quiero murir")
