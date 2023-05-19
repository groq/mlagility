# labels: test_group::monthly author::Narrativa name::bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization downloads::313 task::Natural_Language_Processing sub_task::Summarization
import torch
from transformers import RobertaTokenizerFast, EncoderDecoderModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = 'Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization'
tokenizer = RobertaTokenizerFast.from_pretrained(ckpt)
model = EncoderDecoderModel.from_pretrained(ckpt).to(device)

def generate_summary(text):

   inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
   input_ids = inputs.input_ids.to(device)
   attention_mask = inputs.attention_mask.to(device)
   output = model.generate(input_ids, attention_mask=attention_mask)
   return tokenizer.decode(output[0], skip_special_tokens=True)
   
text = "Your text here..."
generate_summary(text)
