# labels: test_group::monthly author::sentence-transformers name::stsb-roberta-large downloads::8,685 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/stsb-roberta-large')
embeddings = model.encode(sentences)
print(embeddings)
