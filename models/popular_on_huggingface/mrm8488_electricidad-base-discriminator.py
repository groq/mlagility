# labels: test_group::monthly author::mrm8488 name::electricidad-base-discriminator task::Natural_Language_Processing downloads::656
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch

discriminator = ElectraForPreTraining.from_pretrained("mrm8488/electricidad-base-discriminator")
tokenizer = ElectraTokenizerFast.from_pretrained("mrm8488/electricidad-base-discriminator")

sentence = "El rápido zorro marrón salta sobre el perro perezoso"
fake_sentence = "El rápido zorro marrón amar sobre el perro perezoso"

fake_tokens = tokenizer.tokenize(fake_sentence)
fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
discriminator_outputs = discriminator(fake_inputs)
predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

[print("%7s" % token, end="") for token in fake_tokens]

[print("%7s" % prediction, end="") for prediction in predictions.tolist()]

# Output:
'''
el rapido  zorro  marro    ##n   amar  sobre     el  perro   pere ##zoso    0.0    0.0    0.0    0.0    0.0    0.0    1.0    1.0    0.0    0.0    0.0    0.0    0.0[None, None, None, None, None, None, None, None, None, None, None, None, None
'''
