# labels: test_group::monthly author::hakurei name::lit-6B downloads::2,564 license::mit task::Natural_Language_Processing sub_task::Text_Generation
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('hakurei/lit-6B')
tokenizer = AutoTokenizer.from_pretrained('hakurei/lit-6B')

prompt = '''[ Title: The Dunwich Horror; Author: H. P. Lovecraft; Genre: Horror ]
***
When a traveler'''

input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, do_sample=True, temperature=1.0, top_p=0.9, repetition_penalty=1.2, max_length=len(input_ids[0])+100, pad_token_id=tokenizer.eos_token_id)

generated_text = tokenizer.decode(output[0])
print(generated_text)
