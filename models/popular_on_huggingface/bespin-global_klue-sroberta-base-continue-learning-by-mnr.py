# labels: test_group::monthly author::bespin-global name::klue-sroberta-base-continue-learning-by-mnr downloads::785 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer("bespin-global/klue-sroberta-base-continue-learning-by-mnr")
embeddings = model.encode(sentences)
print(embeddings)
