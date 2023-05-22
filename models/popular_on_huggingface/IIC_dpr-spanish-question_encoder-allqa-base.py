# labels: test_group::monthly author::IIC name::dpr-spanish-question_encoder-allqa-base downloads::391 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

model_str = "IIC/dpr-spanish-question_encoder-allqa-base"
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_str)
model = DPRQuestionEncoder.from_pretrained(model_str)

input_ids = tokenizer("¿Qué medallas ganó Usain Bolt en 2012?", return_tensors="pt")["input_ids"]
embeddings = model(input_ids).pooler_output
