# labels: test_group::monthly author::voidful name::dpr-question_encoder-bert-base-multilingual downloads::1,963 task::Multimodal sub_task::Feature_Extraction
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('voidful/dpr-question_encoder-bert-base-multilingual')
model = DPRQuestionEncoder.from_pretrained('voidful/dpr-question_encoder-bert-base-multilingual')
input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='pt')["input_ids"]
embeddings = model(input_ids).pooler_output
