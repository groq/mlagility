# labels: test_group::monthly author::setu4993 name::smaller-LaBSE downloads::1,111 license::apache-2.0 task::Multimodal sub_task::Feature_Extraction
import torch
from transformers import BertModel, BertTokenizerFast


tokenizer = BertTokenizerFast.from_pretrained("setu4993/smaller-LaBSE")
model = BertModel.from_pretrained("setu4993/smaller-LaBSE")
model = model.eval()

english_sentences = [
    "dog",
    "Puppies are nice.",
    "I enjoy taking long walks along the beach with my dog.",
]
english_inputs = tokenizer(english_sentences, return_tensors="pt", padding=True)

with torch.no_grad():
    english_outputs = model(**english_inputs)
