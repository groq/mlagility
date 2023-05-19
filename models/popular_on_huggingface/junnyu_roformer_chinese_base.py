# labels: test_group::monthly author::junnyu name::roformer_chinese_base downloads::3,143 task::Natural_Language_Processing sub_task::Fill-Mask
import torch
from transformers import RoFormerForMaskedLM, RoFormerTokenizer

text = "今天[MASK]很好，我想去公园玩！"
tokenizer = RoFormerTokenizer.from_pretrained("junnyu/roformer_chinese_base")
pt_model = RoFormerForMaskedLM.from_pretrained("junnyu/roformer_chinese_base")
pt_inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    pt_outputs = pt_model(**pt_inputs).logits[0]
pt_outputs_sentence = "pytorch: "
for i, id in enumerate(tokenizer.encode(text)):
    if id == tokenizer.mask_token_id:
        tokens = tokenizer.convert_ids_to_tokens(pt_outputs[i].topk(k=5)[1])
        pt_outputs_sentence += "[" + "||".join(tokens) + "]"
    else:
        pt_outputs_sentence += "".join(
            tokenizer.convert_ids_to_tokens([id], skip_special_tokens=True))
print(pt_outputs_sentence)
# pytorch: 今天[天气||天||阳光||太阳||空气]很好，我想去公园玩！
