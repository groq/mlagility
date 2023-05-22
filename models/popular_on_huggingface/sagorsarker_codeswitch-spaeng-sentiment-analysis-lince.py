# labels: test_group::monthly author::sagorsarker name::codeswitch-spaeng-sentiment-analysis-lince downloads::236 license::mit task::Natural_Language_Processing sub_task::Text_Classification

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-spaeng-sentiment-analysis-lince")

model = AutoModelForSequenceClassification.from_pretrained("sagorsarker/codeswitch-spaeng-sentiment-analysis-lince")

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
sentence = "El perro le ladraba a La Gatita .. .. lol #teamlagatita en las playas de Key Biscayne este Memorial day"
nlp(sentence)
