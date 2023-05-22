# labels: test_group::monthly author::KES name::T5-TTParser downloads::1,381 license::cc-by-nc-sa-4.0 task::Natural_Language_Processing sub_task::Text2Text_Generation

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("KES/T5-TTParser")

model = AutoModelForSeq2SeqLM.from_pretrained("KES/T5-TTParser")

txt = "Ah have live with mi paremnts en London"
inputs = tokenizer("grammar:"+txt, truncation=True, return_tensors='pt')

output = model.generate(inputs['input_ids'], num_beams=4, max_length=512, early_stopping=True)
correction=tokenizer.batch_decode(output, skip_special_tokens=True)
print("".join(correction)) #Correction: Ah live with meh parents in London.
