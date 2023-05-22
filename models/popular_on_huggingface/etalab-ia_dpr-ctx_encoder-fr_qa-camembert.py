# labels: test_group::monthly author::etalab-ia name::dpr-ctx_encoder-fr_qa-camembert task::Natural_Language_Processing downloads::2,213
from transformers import AutoTokenizer, AutoModel
query = "Salut, mon chien est-il mignon ?"
tokenizer = AutoTokenizer.from_pretrained("etalab-ia/dpr-ctx_encoder-fr_qa-camembert",  do_lower_case=True)
input_ids = tokenizer(query, return_tensors='pt')["input_ids"]
model = AutoModel.from_pretrained("etalab-ia/dpr-ctx_encoder-fr_qa-camembert", return_dict=True)
embeddings = model.forward(input_ids).pooler_output
print(embeddings)
