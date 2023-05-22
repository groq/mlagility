# labels: test_group::monthly author::Musixmatch name::umberto-commoncrawl-cased-v1 downloads::3,557 task::Natural_Language_Processing sub_task::Fill-Mask

import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Musixmatch/umberto-commoncrawl-cased-v1")
umberto = AutoModel.from_pretrained("Musixmatch/umberto-commoncrawl-cased-v1")

encoded_input = tokenizer.encode("Umberto Eco è stato un grande scrittore")
input_ids = torch.tensor(encoded_input).unsqueeze(0)  # Batch size 1
outputs = umberto(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output
