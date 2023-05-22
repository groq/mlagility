# labels: test_group::monthly author::sentence-transformers name::nli-mpnet-base-v2 downloads::62,379 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/nli-mpnet-base-v2')
embeddings = model.encode(sentences)
print(embeddings)
