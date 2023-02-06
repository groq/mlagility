# labels: test_group::monthly author::sentence-transformers name::all-roberta-large-v1 downloads::12,097 license::apache-2.0 task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
embeddings = model.encode(sentences)
print(embeddings)
