# labels: test_group::monthly author::bespin-global name::klue-bert-base-mrc downloads::1,844 license::cc-by-nc-4.0 task::Natural_Language_Processing sub_task::Question_Answering
# Load Transformers library
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

context = "your context"
question = "your question"

# Load fine-tuned MRC model by HuggingFace Model Hub
HUGGINGFACE_MODEL_PATH = "bespin-global/klue-bert-base-mrc"
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH )
model = AutoModelForQuestionAnswering.from_pretrained(HUGGINGFACE_MODEL_PATH )

# Encoding
encodings = tokenizer(context, question, 
                      max_length=512, 
                      truncation=True,
                      padding="max_length", 
                      return_token_type_ids=False
                      )
encodings = {key: torch.tensor([val]) for key, val in encodings.items()}             
input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]

# Predict
pred = model(input_ids, attention_mask=attention_mask)

start_logits, end_logits = pred.start_logits, pred.end_logits
token_start_index, token_end_index = start_logits.argmax(dim=-1), end_logits.argmax(dim=-1)
pred_ids = input_ids[0][token_start_index: token_end_index + 1]

# Decoding
prediction = tokenizer.decode(pred_ids)
