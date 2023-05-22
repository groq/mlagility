# labels: test_group::monthly author::sentence-transformers name::paraphrase-albert-small-v2 downloads::21,188 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/paraphrase-albert-small-v2')
embeddings = model.encode(sentences)
print(embeddings)
