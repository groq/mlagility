# labels: test_group::monthly author::sentence-transformers name::gtr-t5-base downloads::486 license::apache-2.0 task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/gtr-t5-base')
embeddings = model.encode(sentences)
print(embeddings)
