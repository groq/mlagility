# labels: test_group::monthly author::sentence-transformers name::use-cmlm-multilingual downloads::226 license::apache-2.0 task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/use-cmlm-multilingual')
embeddings = model.encode(sentences)
print(embeddings)
