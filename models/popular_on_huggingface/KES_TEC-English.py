# labels: test_group::monthly author::KES name::TEC-English downloads::7,358 license::apache-2.0 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("KES/TEC-English")

model = AutoModelForSeq2SeqLM.from_pretrained("KES/TEC-English")
text = "Dem men doh kno wat dey doing wid d money"
inputs = tokenizer("tec:"+text, truncation=True, return_tensors='pt')

output = model.generate(inputs['input_ids'], num_beams=4, max_length=512, early_stopping=True)
translation=tokenizer.batch_decode(output, skip_special_tokens=True)
print("".join(translation)) #translation: These men do not know what they are doing with the money.
