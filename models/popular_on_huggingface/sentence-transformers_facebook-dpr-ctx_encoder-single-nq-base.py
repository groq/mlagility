# labels: test_group::monthly author::sentence-transformers name::facebook-dpr-ctx_encoder-single-nq-base downloads::208 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base')
embeddings = model.encode(sentences)
print(embeddings)
