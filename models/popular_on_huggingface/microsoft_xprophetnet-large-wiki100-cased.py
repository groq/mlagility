# labels: test_group::monthly,daily author::microsoft name::xprophetnet-large-wiki100-cased downloads::540 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import XLMProphetNetForConditionalGeneration, XLMProphetNetTokenizer

model = XLMProphetNetForConditionalGeneration.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")
tokenizer = XLMProphetNetTokenizer.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")

input_str = "the us state department said wednesday it had received no formal word from bolivia that it was expelling the us ambassador there but said the charges made against him are `` baseless ."
target_str = "us rejects charges against its ambassador in bolivia"

input_ids = tokenizer(input_str, return_tensors="pt").input_ids
labels = tokenizer(target_str, return_tensors="pt").input_ids

loss = model(input_ids, labels=labels).loss
