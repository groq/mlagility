# labels: test_group::monthly author::laituan245 name::molt5-large-smiles2caption downloads::185 license::apache-2.0 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-large-smiles2caption", model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-large-smiles2caption')

input_text = 'C1=CC2=C(C(=C1)[O-])NC(=CC2=O)C(=O)O'
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids, num_beams=5, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
