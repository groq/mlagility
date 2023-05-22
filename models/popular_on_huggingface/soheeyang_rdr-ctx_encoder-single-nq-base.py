# labels: test_group::monthly author::soheeyang name::rdr-ctx_encoder-single-nq-base task::Natural_Language_Processing downloads::1,110
from transformers import DPRContextEncoder, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("soheeyang/rdr-ctx_encoder-single-nq-base")
ctx_encoder = DPRContextEncoder.from_pretrained("soheeyang/rdr-ctx_encoder-single-nq-base")

data = tokenizer("context comes here", return_tensors="pt")
ctx_embedding = ctx_encoder(**data).pooler_output  # embedding vector for context
