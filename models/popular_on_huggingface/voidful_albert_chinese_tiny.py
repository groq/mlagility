# labels: test_group::monthly author::voidful name::albert_chinese_tiny downloads::1,127 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import AutoTokenizer, AlbertForMaskedLM
import torch
from torch.nn.functional import softmax

pretrained = 'voidful/albert_chinese_tiny'
tokenizer = AutoTokenizer.from_pretrained(pretrained)
model = AlbertForMaskedLM.from_pretrained(pretrained)

inputtext = "今天[MASK]情很好"

maskpos = tokenizer.encode(inputtext, add_special_tokens=True).index(103)

input_ids = torch.tensor(tokenizer.encode(inputtext, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=input_ids)
loss, prediction_scores = outputs[:2]
logit_prob = softmax(prediction_scores[0, maskpos],dim=-1).data.tolist()
predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token, logit_prob[predicted_index])
