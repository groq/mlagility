# labels: test_group::monthly author::snunlp name::KR-SBERT-V40K-klueNLI-augSTS downloads::1,446 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
embeddings = model.encode(sentences)
print(embeddings)
