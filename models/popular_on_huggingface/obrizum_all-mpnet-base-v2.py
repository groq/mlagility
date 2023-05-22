# labels: test_group::monthly author::obrizum name::all-mpnet-base-v2 downloads::186 license::apache-2.0 task::Multimodal sub_task::Feature_Extraction
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('obrizum/all-mpnet-base-v2')
embeddings = model.encode(sentences)
print(embeddings)
