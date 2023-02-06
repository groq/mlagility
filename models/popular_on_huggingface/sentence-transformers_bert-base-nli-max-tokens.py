# labels: test_group::monthly author::sentence-transformers name::bert-base-nli-max-tokens downloads::664 license::apache-2.0 task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/bert-base-nli-max-tokens')
embeddings = model.encode(sentences)
print(embeddings)
