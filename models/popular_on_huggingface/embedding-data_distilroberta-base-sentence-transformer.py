# labels: test_group::monthly author::embedding-data name::distilroberta-base-sentence-transformer downloads::220 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('embedding-data/distilroberta-base-sentence-transformer')
embeddings = model.encode(sentences)
print(embeddings)
