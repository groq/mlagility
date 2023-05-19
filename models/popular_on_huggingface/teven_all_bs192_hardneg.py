# labels: test_group::monthly author::teven name::all_bs192_hardneg downloads::428 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('teven/all_bs192_hardneg')
embeddings = model.encode(sentences)
print(embeddings)
