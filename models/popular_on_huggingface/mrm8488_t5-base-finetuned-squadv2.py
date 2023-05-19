# labels: test_group::monthly author::mrm8488 name::t5-base-finetuned-squadv2 downloads::215 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-squadv2")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-squadv2")

def get_answer(question, context):
  input_text = "question: %s  context: %s" % (question, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'])
  
  return tokenizer.decode(output[0])

context = "Manuel have created RuPERTa-base with the support of HF-Transformers and Google"
question = "Who has supported Manuel?"

get_answer(question, context)

# output: 'HF-Transformers and Google'
