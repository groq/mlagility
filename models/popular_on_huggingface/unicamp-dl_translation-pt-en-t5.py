# labels: test_group::monthly author::unicamp-dl name::translation-pt-en-t5 downloads::666 task::Translation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
  
tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/translation-pt-en-t5")

model = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-pt-en-t5")

pten_pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

pten_pipeline("translate Portuguese to English: Eu gosto de comer arroz.")
