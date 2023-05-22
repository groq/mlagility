# labels: test_group::monthly author::pritamdeka name::S-BioBert-snli-multinli-stsb downloads::584 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('pritamdeka/S-BioBert-snli-multinli-stsb')
embeddings = model.encode(sentences)
print(embeddings)
