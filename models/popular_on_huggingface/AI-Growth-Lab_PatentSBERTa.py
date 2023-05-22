# labels: test_group::monthly author::AI-Growth-Lab name::PatentSBERTa downloads::2,644 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
embeddings = model.encode(sentences)
print(embeddings)
