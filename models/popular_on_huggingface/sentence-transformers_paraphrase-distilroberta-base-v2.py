# labels: test_group::monthly author::sentence-transformers name::paraphrase-distilroberta-base-v2 downloads::28,681 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v2')
embeddings = model.encode(sentences)
print(embeddings)
