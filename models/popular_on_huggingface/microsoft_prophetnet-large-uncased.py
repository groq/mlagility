# labels: test_group::monthly,daily author::microsoft name::prophetnet-large-uncased downloads::5,629 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer

model = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased")
tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")

input_str = "the us state department said wednesday it had received no formal word from bolivia that it was expelling the us ambassador there but said the charges made against him are `` baseless ."
target_str = "us rejects charges against its ambassador in bolivia"

input_ids = tokenizer(input_str, return_tensors="pt").input_ids
labels = tokenizer(target_str, return_tensors="pt").input_ids

loss = model(input_ids, labels=labels).loss
