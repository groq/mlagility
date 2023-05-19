# labels: test_group::monthly author::sentence-transformers name::msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned downloads::474 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned')
embeddings = model.encode(sentences)
print(embeddings)
