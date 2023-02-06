# labels: test_group::monthly author::deepset name::all-mpnet-base-v2-table downloads::7,007 task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('deepset/all-mpnet-base-v2-table')
embeddings = model.encode(sentences)
print(embeddings)
