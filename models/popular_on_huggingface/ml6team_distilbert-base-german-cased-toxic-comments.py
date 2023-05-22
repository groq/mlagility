# labels: test_group::monthly author::ml6team name::distilbert-base-german-cased-toxic-comments downloads::861 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline

model_hub_url = 'https://huggingface.co/ml6team/distilbert-base-german-cased-toxic-comments'
model_name = 'ml6team/distilbert-base-german-cased-toxic-comments'

toxicity_pipeline = pipeline('text-classification', model=model_name, tokenizer=model_name)

comment = "Ein harmloses Beispiel"
result = toxicity_pipeline(comment)[0]
print(f"Comment: {comment}\nLabel: {result['label']}, score: {result['score']}")
