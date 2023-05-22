# labels: test_group::monthly author::Rostlab name::prot_t5_xl_uniref50 downloads::26,060 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import T5Tokenizer, T5Model
import re
import torch

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)

model = T5Model.from_pretrained("Rostlab/prot_t5_xl_uniref50")

sequences_Example = ["A E T C Z A O","S K T Z P"]

sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)

input_ids = torch.tensor(ids['input_ids'])
attention_mask = torch.tensor(ids['attention_mask'])

with torch.no_grad():
    embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)

# For feature extraction we recommend to use the encoder embedding
encoder_embedding = embedding[2].cpu().numpy()
decoder_embedding = embedding[0].cpu().numpy()
