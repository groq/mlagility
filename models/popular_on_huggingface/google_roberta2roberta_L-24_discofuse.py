# labels: test_group::monthly author::google name::roberta2roberta_L-24_discofuse downloads::268 license::apache-2.0 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")
model = AutoModelForSeq2SeqLM.from_pretrained("google/roberta2roberta_L-24_discofuse")

discofuse = """As a run-blocker, Zeitler moves relatively well. Zeitler often struggles at the point of contact in space."""

input_ids = tokenizer(discofuse, return_tensors="pt").input_ids
output_ids = model.generate(input_ids)[0]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
# should output
# As a run-blocker, Zeitler moves relatively well. However, Zeitler often struggles at the point of contact in space.  
