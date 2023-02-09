# labels: test_group::monthly author::sentence-transformers name::allenai-specter downloads::1,458 license::apache-2.0 task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/allenai-specter')
embeddings = model.encode(sentences)
print(embeddings)
