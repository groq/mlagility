# labels: test_group::monthly author::google name::long-t5-tglobal-xl downloads::984 license::apache-2.0 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import AutoTokenizer, LongT5Model

tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-xl")
model = LongT5Model.from_pretrained("google/long-t5-tglobal-xl")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
