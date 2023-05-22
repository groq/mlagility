# labels: test_group::monthly author::pritamdeka name::S-PubMedBert-MS-MARCO downloads::352 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
embeddings = model.encode(sentences)
print(embeddings)
