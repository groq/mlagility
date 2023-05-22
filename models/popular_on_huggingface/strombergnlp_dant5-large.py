# labels: test_group::monthly author::strombergnlp name::dant5-large downloads::332 license::cc-by-4.0 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import AutoTokenizer, T5ForConditionalGeneration

model_name = "strombergnlp/dant5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

original_text = "Aarhus er Danmarks <extra_id_0> landets ældste. Under navnet Aros, som betyder å-munding, optræder den i skriftlige kilder i 900-tallet, men <extra_id_1> historie tilbage til 700-tallet.<extra_id_2>"
original_label = "<extra_id_0> næststørste by og en af <extra_id_1> arkæologiske fund fører dens <extra_id_2>"
input_ids = tokenizer(original_text, return_tensors="pt").input_ids
labels = tokenizer(original_label, return_tensors="pt").input_ids

loss = model(input_ids=input_ids, labels=labels).loss
print(f"Original text: {original_text}")
print(f"Original label: {original_label}")
print(f"Loss for the original label is {loss.item()}")

sequence_ids = model.generate(input_ids)
sequences = tokenizer.batch_decode(sequence_ids)
print(f"A sample generated continuation: ")
print(sequences[0])
