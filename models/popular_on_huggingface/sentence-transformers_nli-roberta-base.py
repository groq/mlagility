# labels: test_group::monthly author::sentence-transformers name::nli-roberta-base downloads::3,190 license::apache-2.0 task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/nli-roberta-base')
embeddings = model.encode(sentences)
print(embeddings)
