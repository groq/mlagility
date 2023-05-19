# labels: test_group::monthly author::etalab-ia name::dpr-question_encoder-fr_qa-camembert downloads::2,145 task::Multimodal sub_task::Feature_Extraction
from transformers import AutoTokenizer, AutoModel
query = "Salut, mon chien est-il mignon ?"
tokenizer = AutoTokenizer.from_pretrained("etalab-ia/dpr-question_encoder-fr_qa-camembert",  do_lower_case=True)
input_ids = tokenizer(query, return_tensors='pt')["input_ids"]
model = AutoModel.from_pretrained("etalab-ia/dpr-question_encoder-fr_qa-camembert", return_dict=True)
embeddings = model.forward(input_ids).pooler_output
print(embeddings)
