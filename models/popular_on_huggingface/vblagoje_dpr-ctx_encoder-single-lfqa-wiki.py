# labels: test_group::monthly author::vblagoje name::dpr-ctx_encoder-single-lfqa-wiki task::Natural_Language_Processing downloads::8,721 license::mit
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

tokenizer = DPRContextEncoderTokenizer.from_pretrained("vblagoje/dpr-ctx_encoder-single-lfqa-wiki")
model = DPRContextEncoder.from_pretrained("vblagoje/dpr-ctx_encoder-single-lfqa-wiki")
input_ids = tokenizer("Where an aircraft passes through a cloud, it can disperse the cloud in its path...", return_tensors="pt")["input_ids"]
embeddings = model(input_ids).pooler_output
