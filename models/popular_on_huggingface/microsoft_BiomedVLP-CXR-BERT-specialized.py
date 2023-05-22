# labels: test_group::monthly author::microsoft name::BiomedVLP-CXR-BERT-specialized downloads::5,824 license::mit task::Natural_Language_Processing sub_task::Fill-Mask
import torch
from transformers import AutoModel, AutoTokenizer

# Load the model and tokenizer
url = "microsoft/BiomedVLP-CXR-BERT-specialized"
tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
model = AutoModel.from_pretrained(url, trust_remote_code=True)

# Input text prompts (e.g., reference, synonym, contradiction)
text_prompts = ["There is no pneumothorax or pleural effusion",
                "No pleural effusion or pneumothorax is seen",
                "The extent of the pleural effusion is constant."]

# Tokenize and compute the sentence embeddings
tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=text_prompts,
                                               add_special_tokens=True,
                                               padding='longest',
                                               return_tensors='pt')
embeddings = model.get_projected_text_embeddings(input_ids=tokenizer_output.input_ids,
                                                 attention_mask=tokenizer_output.attention_mask)

# Compute the cosine similarity of sentence embeddings obtained from input text prompts.
sim = torch.mm(embeddings, embeddings.t())
