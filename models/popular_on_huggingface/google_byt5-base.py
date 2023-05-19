# labels: test_group::monthly,daily author::google name::byt5-base downloads::3,256 license::apache-2.0 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import T5ForConditionalGeneration
import torch

model = T5ForConditionalGeneration.from_pretrained('google/byt5-base')

input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + 3  # add 3 for special tokens
labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + 3  # add 3 for special tokens

loss = model(input_ids, labels=labels).loss # forward pass
