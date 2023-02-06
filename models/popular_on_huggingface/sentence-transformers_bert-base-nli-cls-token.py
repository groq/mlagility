# labels: test_group::monthly author::sentence-transformers name::bert-base-nli-cls-token downloads::1,313 license::apache-2.0 task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/bert-base-nli-cls-token')
embeddings = model.encode(sentences)
print(embeddings)
