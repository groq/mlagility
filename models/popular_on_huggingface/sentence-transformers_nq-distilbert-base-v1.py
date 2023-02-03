# labels: test_group::monthly author::sentence-transformers name::nq-distilbert-base-v1 downloads::3,766 license::apache-2.0 task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/nq-distilbert-base-v1')
embeddings = model.encode(sentences)
print(embeddings)
