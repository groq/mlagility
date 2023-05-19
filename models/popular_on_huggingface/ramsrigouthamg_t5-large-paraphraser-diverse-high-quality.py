# labels: test_group::monthly author::ramsrigouthamg name::t5-large-paraphraser-diverse-high-quality downloads::69,043 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)

# Beam Search

context = "Once, a group of frogs were roaming around the forest in search of water."
text = "paraphrase: "+context + " </s>"

encoding = tokenizer.encode_plus(text,max_length =128, padding=True, return_tensors="pt")
input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

model.eval()
beam_outputs = model.generate(
    input_ids=input_ids,attention_mask=attention_mask,
    max_length=128,
    early_stopping=True,
    num_beams=15,
    num_return_sequences=3

)

print ("\n\n")
print ("Original: ",context)
for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print (sent)
