# labels: test_group::monthly author::valhalla name::t5-base-squad downloads::99,402 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-squad")
model = AutoModelWithLMHead.from_pretrained("valhalla/t5-base-squad")

def get_answer(question, context):
  input_text = "question: %s  context: %s </s>" % (question, context)
  features = tokenizer([input_text], return_tensors='pt')

  out = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'])
  
  return tokenizer.decode(out[0])

context = "In Norse mythology, Valhalla is a majestic, enormous hall located in Asgard, ruled over by the god Odin."
question = "What is Valhalla ?"

get_answer(question, context)
# output: 'a majestic, enormous hall located in Asgard, ruled over by the god Odin'
