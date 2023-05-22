# labels: test_group::monthly author::nbroad name::mt5-base-qgen downloads::216 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("nbroad/mt5-base-qgen")
model = AutoModelForSeq2SeqLM.from_pretrained("nbroad/mt5-base-qgen")

text = "Hugging Face has seen rapid growth in its \
popularity since the get-go. It is definitely doing\
 the right things to attract more and more people to \
 its platform, some of which are on the following lines:\
Community driven approach through large open source repositories \
along with paid services. Helps to build a network of like-minded\
 people passionate about open source. \
Attractive price point. The subscription-based features, e.g.: \
Inference based API, starts at a price of $9/month.\
"

inputs = tokenizer(text, return_tensors="pt")
output = model.generate(**inputs, max_length=40)

tokenizer.decode(output[0], skip_special_tokens=True)
# What is Hugging Face's price point?
