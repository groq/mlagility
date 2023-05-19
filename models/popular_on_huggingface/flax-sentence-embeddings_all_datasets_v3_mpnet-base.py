# labels: test_group::monthly author::flax-sentence-embeddings name::all_datasets_v3_mpnet-base downloads::269 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v1')
embeddings = model.encode(sentences)
print(embeddings)
