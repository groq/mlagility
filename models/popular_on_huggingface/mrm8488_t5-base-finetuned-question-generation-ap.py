# labels: test_group::monthly author::mrm8488 name::t5-base-finetuned-question-generation-ap downloads::882,467 task::Natural_Language_Processing sub_task::Text2Text_Generation
# Tip: By now, install transformers from source

from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

def get_question(answer, context, max_length=64):
  input_text = "answer: %s  context: %s </s>" % (answer, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer.decode(output[0])

context = "Manuel has created RuPERTa-base with the support of HF-Transformers and Google"
answer = "Manuel"

get_question(answer, context)

# output: question: Who created the RuPERTa-base?
