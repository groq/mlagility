# labels: test_group::monthly author::voidful name::dpr-ctx_encoder-bert-base-multilingual task::Natural_Language_Processing downloads::1,900
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
tokenizer = DPRContextEncoderTokenizer.from_pretrained('voidful/dpr-ctx_encoder-bert-base-multilingual')
model = DPRContextEncoder.from_pretrained('voidful/dpr-ctx_encoder-bert-base-multilingual')
input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='pt')["input_ids"]
embeddings = model(input_ids).pooler_output
