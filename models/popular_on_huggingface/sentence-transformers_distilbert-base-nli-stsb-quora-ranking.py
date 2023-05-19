# labels: test_group::monthly author::sentence-transformers name::distilbert-base-nli-stsb-quora-ranking downloads::570 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-quora-ranking')
embeddings = model.encode(sentences)
print(embeddings)
