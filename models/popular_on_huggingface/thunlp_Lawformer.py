# labels: test_group::monthly author::thunlp name::Lawformer downloads::2,116 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("thunlp/Lawformer")
model = AutoModel.from_pretrained("thunlp/Lawformer")
inputs = tokenizer("任某提起诉讼，请求判令解除婚姻关系并对夫妻共同财产进行分割。", return_tensors="pt")
outputs = model(**inputs)

