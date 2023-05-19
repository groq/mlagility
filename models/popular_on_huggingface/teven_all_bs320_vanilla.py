# labels: test_group::monthly author::teven name::all_bs320_vanilla downloads::426 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('teven/all_bs320_vanilla')
embeddings = model.encode(sentences)
print(embeddings)
