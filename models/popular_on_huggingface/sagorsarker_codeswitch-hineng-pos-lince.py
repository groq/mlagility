# labels: test_group::monthly author::sagorsarker name::codeswitch-hineng-pos-lince downloads::237 license::mit task::Natural_Language_Processing sub_task::Token_Classification

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-hineng-pos-lince")

model = AutoModelForTokenClassification.from_pretrained("sagorsarker/codeswitch-hineng-pos-lince")
pos_model = pipeline('ner', model=model, tokenizer=tokenizer)

pos_model("put any hindi english code-mixed sentence")
