# labels: test_group::monthly author::sagorsarker name::codeswitch-hineng-lid-lince downloads::310 license::mit task::Natural_Language_Processing sub_task::Token_Classification

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-hineng-lid-lince")

model = AutoModelForTokenClassification.from_pretrained("sagorsarker/codeswitch-hineng-lid-lince")
lid_model = pipeline('ner', model=model, tokenizer=tokenizer)

lid_model("put any hindi english code-mixed sentence")
