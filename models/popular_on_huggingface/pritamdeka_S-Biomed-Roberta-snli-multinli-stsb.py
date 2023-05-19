# labels: test_group::monthly author::pritamdeka name::S-Biomed-Roberta-snli-multinli-stsb downloads::9,793 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('pritamdeka/S-Biomed-Roberta-snli-multinli-stsb')
embeddings = model.encode(sentences)
print(embeddings)
