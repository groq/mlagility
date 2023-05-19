# labels: test_group::monthly author::pin name::senda downloads::27,194 license::cc-by-4.0 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
tokenizer = AutoTokenizer.from_pretrained("pin/senda")
model = AutoModelForSequenceClassification.from_pretrained("pin/senda")

# create 'senda' sentiment analysis pipeline 
senda_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

text = "Sikke en dejlig dag det er i dag"
# in English: 'what a lovely day'
senda_pipeline(text)
