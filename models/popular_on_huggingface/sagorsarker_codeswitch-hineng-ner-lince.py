# labels: test_group::monthly author::sagorsarker name::codeswitch-hineng-ner-lince downloads::7,147 license::mit task::Natural_Language_Processing sub_task::Token_Classification

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-hineng-ner-lince")

model = AutoModelForTokenClassification.from_pretrained("sagorsarker/codeswitch-hineng-ner-lince")

ner_model = pipeline('ner', model=model, tokenizer=tokenizer)

ner_model("put any hindi english code-mixed sentence")
