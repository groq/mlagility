# labels: test_group::monthly author::sentence-transformers name::paraphrase-TinyBERT-L6-v2 downloads::24,933 license::apache-2.0 task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/paraphrase-TinyBERT-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
