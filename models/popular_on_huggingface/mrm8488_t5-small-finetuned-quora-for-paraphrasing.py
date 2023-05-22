# labels: test_group::monthly author::mrm8488 name::t5-small-finetuned-quora-for-paraphrasing downloads::408 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")

def paraphrase(text, max_length=128):

  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

  generated_ids = model.generate(input_ids=input_ids, num_return_sequences=5, num_beams=5, max_length=max_length, no_repeat_ngram_size=2, repetition_penalty=3.5, length_penalty=1.0, early_stopping=True)

  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

  return preds
  
preds = paraphrase("paraphrase: What is the best framework for dealing with a huge text dataset?")

for pred in preds:
  print(pred)

# Output:
'''
What is the best framework for dealing with a huge text dataset?
What is the best framework for dealing with a large text dataset?
What is the best framework to deal with a huge text dataset?
What are the best frameworks for dealing with a huge text dataset?
What is the best framework for dealing with huge text datasets?
'''
