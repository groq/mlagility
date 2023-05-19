# labels: test_group::monthly author::mrm8488 name::t5-base-finetuned-quartz downloads::254 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-quartz")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-quartz")

def get_response(question, fact, opts, max_length=16):
  input_text = 'question: %s context: %s options: %s' % (question, fact, opts)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer.decode(output[0])
  
fact = 'The sooner cancer is detected the easier it is to treat.'
question = 'John was a doctor in a cancer ward and knew that early detection was key. The cancer being detected quickly makes the cancer treatment'
opts = 'Easier, Harder'

get_response(question, fact, opts)

# output: 'Easier'
