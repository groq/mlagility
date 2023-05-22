# labels: test_group::monthly author::IIC name::dpr-spanish-passage_encoder-allqa-base downloads::388 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

model_str = "IIC/dpr-spanish-passage_encoder-allqa-base"
tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_str)
model = DPRContextEncoder.from_pretrained(model_str)

input_ids = tokenizer("Usain Bolt ganó varias medallas de oro en las Olimpiadas del año 2012", return_tensors="pt")["input_ids"]
embeddings = model(input_ids).pooler_output
