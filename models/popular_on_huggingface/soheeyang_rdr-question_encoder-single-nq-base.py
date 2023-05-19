# labels: test_group::monthly author::soheeyang name::rdr-question_encoder-single-nq-base downloads::1,236 task::Multimodal sub_task::Feature_Extraction
from transformers import DPRQuestionEncoder, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("soheeyang/rdr-question_encoder-single-trivia-base")
question_encoder = DPRQuestionEncoder.from_pretrained("soheeyang/rdr-question_encoder-single-trivia-base")

data = tokenizer("question comes here", return_tensors="pt")
question_embedding = question_encoder(**data).pooler_output  # embedding vector for question
