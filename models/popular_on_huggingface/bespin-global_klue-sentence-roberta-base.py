# labels: test_group::monthly author::bespin-global name::klue-sentence-roberta-base downloads::238 license::cc-by-nc-4.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('bespin-global/klue-sentence-roberta-base')
embeddings = model.encode(sentences)
print(embeddings)
