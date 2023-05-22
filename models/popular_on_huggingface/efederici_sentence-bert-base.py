# labels: test_group::monthly author::efederici name::sentence-bert-base downloads::189 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["Questo è un esempio di frase", "Questo è un ulteriore esempio"]

model = SentenceTransformer('efederici/sentence-bert-base')
embeddings = model.encode(sentences)
print(embeddings)
