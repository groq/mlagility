# labels: test_group::monthly author::sentence-transformers name::gtr-t5-xl downloads::1,037 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/gtr-t5-xl')
embeddings = model.encode(sentences)
print(embeddings)
