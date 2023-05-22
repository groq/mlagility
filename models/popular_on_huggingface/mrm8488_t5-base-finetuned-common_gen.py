# labels: test_group::monthly author::mrm8488 name::t5-base-finetuned-common_gen downloads::451,087 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-common_gen")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-common_gen")

def gen_sentence(words, max_length=32):
  input_text = words
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer.decode(output[0], skip_special_tokens=True)

words = "tree plant ground hole dig"

gen_sentence(words)

# output: digging a hole in the ground to plant trees
