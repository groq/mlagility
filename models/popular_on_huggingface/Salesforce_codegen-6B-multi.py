# labels: test_group::monthly author::Salesforce name::codegen-6B-multi downloads::7,048 license::bsd-3-clause task::Natural_Language_Processing sub_task::Text_Generation
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-6B-multi")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-6B-multi")

text = "def hello_world():"
input_ids = tokenizer(text, return_tensors="pt").input_ids

generated_ids = model.generate(input_ids, max_length=128)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
