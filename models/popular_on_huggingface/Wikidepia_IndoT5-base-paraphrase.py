# labels: test_group::monthly author::Wikidepia name::IndoT5-base-paraphrase downloads::2,254 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Wikidepia/IndoT5-base-paraphrase")  
model = AutoModelForSeq2SeqLM.from_pretrained("Wikidepia/IndoT5-base-paraphrase")

sentence = "Anak anak melakukan piket kelas agar kebersihan kelas terjaga"
text =  "paraphrase: " + sentence + " </s>"

encoding = tokenizer(text, padding='longest', return_tensors="pt")
outputs = model.generate(
    input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"],
    max_length=512,
    do_sample=True,
    top_k=200,
    top_p=0.95,
    early_stopping=True,
    num_return_sequences=5
)
