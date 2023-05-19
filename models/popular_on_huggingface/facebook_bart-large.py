# labels: test_group::monthly,daily author::facebook name::bart-large downloads::523,031 license::apache-2.0 task::Multimodal sub_task::Feature_Extraction
from transformers import BartTokenizer, BartModel

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartModel.from_pretrained('facebook/bart-large')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
