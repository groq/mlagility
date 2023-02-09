# labels: test_group::monthly author::sentence-transformers name::stsb-roberta-base-v2 downloads::103,591 license::apache-2.0 task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/stsb-roberta-base-v2')
embeddings = model.encode(sentences)
print(embeddings)
