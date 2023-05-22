# labels: test_group::monthly author::aubmindlab name::araelectra-base-discriminator task::Natural_Language_Processing downloads::1,266
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch

discriminator = ElectraForPreTraining.from_pretrained("aubmindlab/araelectra-base-discriminator")
tokenizer = ElectraTokenizerFast.from_pretrained("aubmindlab/araelectra-base-discriminator")

sentence = ""
fake_sentence = ""

fake_tokens = tokenizer.tokenize(fake_sentence)
fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
discriminator_outputs = discriminator(fake_inputs)
predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

[print("%7s" % token, end="") for token in fake_tokens]

[print("%7s" % int(prediction), end="") for prediction in predictions.tolist()]
