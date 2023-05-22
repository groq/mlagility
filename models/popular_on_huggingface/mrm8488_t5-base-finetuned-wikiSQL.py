# labels: test_group::monthly author::mrm8488 name::t5-base-finetuned-wikiSQL downloads::5,704 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")

def get_sql(query):
  input_text = "translate English to SQL: %s </s>" % query
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'])
  
  return tokenizer.decode(output[0])

query = "How many models were finetuned using BERT as base model?"

get_sql(query)

# output: 'SELECT COUNT Model fine tuned FROM table WHERE Base model = BERT'
