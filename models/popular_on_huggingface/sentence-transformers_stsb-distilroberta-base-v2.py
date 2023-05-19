# labels: test_group::monthly author::sentence-transformers name::stsb-distilroberta-base-v2 downloads::4,333 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/stsb-distilroberta-base-v2')
embeddings = model.encode(sentences)
print(embeddings)
