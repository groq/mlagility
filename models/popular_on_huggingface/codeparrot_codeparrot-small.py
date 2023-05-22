# labels: test_group::monthly author::codeparrot name::codeparrot-small downloads::2,739 license::apache-2.0 task::Natural_Language_Processing sub_task::Text_Generation
from transformers import AutoTokenizer, AutoModelWithLMHead
  
tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small")
model = AutoModelWithLMHead.from_pretrained("codeparrot/codeparrot-small")

inputs = tokenizer("def hello_world():", return_tensors="pt")
outputs = model(**inputs)
