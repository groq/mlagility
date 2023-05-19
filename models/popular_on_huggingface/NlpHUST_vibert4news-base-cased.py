# labels: test_group::monthly author::NlpHUST name::vibert4news-base-cased downloads::633 task::Natural_Language_Processing sub_task::Fill-Mask
import torch
from transformers import BertTokenizer,BertModel
tokenizer= BertTokenizer.from_pretrained("NlpHUST/vibert4news-base-cased")
bert_model = BertModel.from_pretrained("NlpHUST/vibert4news-base-cased")

line = "Tôi là sinh viên trường Bách Khoa Hà Nội ."
input_id = tokenizer.encode(line,add_special_tokens = True)
att_mask = [int(token_id > 0) for token_id in input_id]
input_ids = torch.tensor([input_id])
att_masks = torch.tensor([att_mask])
with torch.no_grad():
    features = bert_model(input_ids,att_masks)

print(features)
