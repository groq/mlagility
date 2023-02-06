# labels: test_group::monthly author::Salesforce name::codegen-350M-nl downloads::364 license::bsd-3-clause task::Text_Generation
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-nl")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-nl")

text = "def hello_world():"
input_ids = tokenizer(text, return_tensors="pt").input_ids
generated_ids = model.generate(input_ids, max_length=128)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
