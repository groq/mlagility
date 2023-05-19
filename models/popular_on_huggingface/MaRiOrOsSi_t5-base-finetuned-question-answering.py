# labels: test_group::monthly author::MaRiOrOsSi name::t5-base-finetuned-question-answering downloads::845 task::Natural_Language_Processing sub_task::Text2Text_Generation
from  transformers  import  AutoTokenizer, AutoModelWithLMHead, pipeline

model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)
question = "What is 42?"
context = "42 is the answer to life, the universe and everything"
input = f"question: {question} context: {context}"
encoded_input = tokenizer([input],
                             return_tensors='pt',
                             max_length=512,
                             truncation=True)
output = model.generate(input_ids = encoded_input.input_ids,
                            attention_mask = encoded_input.attention_mask)
output = tokenizer.decode(output[0], skip_special_tokens=True)
print(output)
