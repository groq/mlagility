# labels: test_group::monthly author::sentence-transformers name::distiluse-base-multilingual-cased-v2 downloads::1,184,073 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
embeddings = model.encode(sentences)
print(embeddings)
