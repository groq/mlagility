# labels: test_group::monthly author::sentence-transformers name::sentence-t5-xxl downloads::297 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/sentence-t5-xxl')
embeddings = model.encode(sentences)
print(embeddings)
