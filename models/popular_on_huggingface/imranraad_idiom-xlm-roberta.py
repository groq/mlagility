# labels: test_group::monthly author::imranraad name::idiom-xlm-roberta downloads::195 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("imranraad/autotrain-magpie-epie-combine-xlmr-metaphor-1595156286", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("imranraad/autotrain-magpie-epie-combine-xlmr-metaphor-1595156286", use_auth_token=True)

inputs = tokenizer("I love AutoTrain", return_tensors="pt")

outputs = model(**inputs)
