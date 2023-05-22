# labels: test_group::monthly author::ken11 name::albert-base-japanese-v1 downloads::3,207 license::mit task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import (
    AlbertForMaskedLM, AlbertTokenizerFast
)
import torch


tokenizer = AlbertTokenizerFast.from_pretrained("ken11/albert-base-japanese-v1")
model = AlbertForMaskedLM.from_pretrained("ken11/albert-base-japanese-v1")

text = "大学で[MASK]の研究をしています"
tokenized_text = tokenizer.tokenize(text)
del tokenized_text[tokenized_text.index(tokenizer.mask_token) + 1]

input_ids = [tokenizer.cls_token_id]
input_ids.extend(tokenizer.convert_tokens_to_ids(tokenized_text))
input_ids.append(tokenizer.sep_token_id)

inputs = {"input_ids": [input_ids], "token_type_ids": [[0]*len(input_ids)], "attention_mask": [[1]*len(input_ids)]}
batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in inputs.items()}
output = model(**batch)[0]
_, result = output[0, input_ids.index(tokenizer.mask_token_id)].topk(5)

print(tokenizer.convert_ids_to_tokens(result.tolist()))
# ['英語', '心理学', '数学', '医学', '日本語']
