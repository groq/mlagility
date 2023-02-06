# labels: test_group::monthly author::AI-Growth-Lab name::PatentSBERTa downloads::2,644 task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
embeddings = model.encode(sentences)
print(embeddings)
