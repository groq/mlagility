# labels: test_group::monthly author::sentence-transformers name::stsb-bert-large downloads::940 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/stsb-bert-large')
embeddings = model.encode(sentences)
print(embeddings)
