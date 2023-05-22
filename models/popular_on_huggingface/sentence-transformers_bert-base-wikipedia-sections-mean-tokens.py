# labels: test_group::monthly author::sentence-transformers name::bert-base-wikipedia-sections-mean-tokens downloads::2,269 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/bert-base-wikipedia-sections-mean-tokens')
embeddings = model.encode(sentences)
print(embeddings)
