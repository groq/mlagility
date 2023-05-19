# labels: test_group::monthly author::sentence-transformers name::distilbert-base-nli-stsb-mean-tokens downloads::103,186 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
embeddings = model.encode(sentences)
print(embeddings)
