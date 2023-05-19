# labels: test_group::monthly author::KBLab name::sentence-bert-swedish-cased downloads::279 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["Det här är en exempelmening", "Varje exempel blir konverterad"]

model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')
embeddings = model.encode(sentences)
print(embeddings)
