# labels: test_group::monthly author::hiiamsid name::sentence_similarity_spanish_es downloads::3,777 task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
embeddings = model.encode(sentences)
print(embeddings)
