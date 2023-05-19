# labels: test_group::monthly author::sentence-transformers name::distilbert-base-nli-mean-tokens downloads::73,758 license::apache-2.0 task::Multimodal sub_task::Feature_Extraction
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')
embeddings = model.encode(sentences)
print(embeddings)
