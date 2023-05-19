# labels: test_group::monthly author::sentence-transformers name::xlm-r-distilroberta-base-paraphrase-v1 downloads::8,598 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1')
embeddings = model.encode(sentences)
print(embeddings)
