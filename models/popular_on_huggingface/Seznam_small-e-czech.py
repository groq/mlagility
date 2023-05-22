# labels: test_group::monthly author::Seznam name::small-e-czech task::Natural_Language_Processing downloads::8,270 license::cc-by-4.0
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch

discriminator = ElectraForPreTraining.from_pretrained("Seznam/small-e-czech")
tokenizer = ElectraTokenizerFast.from_pretrained("Seznam/small-e-czech")

sentence = "Za hory, za doly, mé zlaté parohy"
fake_sentence = "Za hory, za doly, kočka zlaté parohy"

fake_sentence_tokens = ["[CLS]"] + tokenizer.tokenize(fake_sentence) + ["[SEP]"]
fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
outputs = discriminator(fake_inputs)
predictions = torch.nn.Sigmoid()(outputs[0]).cpu().detach().numpy()

for token in fake_sentence_tokens:
    print("{:>7s}".format(token), end="")
print()

for prediction in predictions.squeeze():
    print("{:7.1f}".format(prediction), end="")
print()
