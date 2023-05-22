# labels: test_group::monthly author::teven name::all_bs160_allneg downloads::433 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('teven/all_bs160_allneg')
embeddings = model.encode(sentences)
print(embeddings)
