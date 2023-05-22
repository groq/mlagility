# labels: test_group::monthly author::hiiamsid name::sentence_similarity_hindi downloads::8,866 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('hiiamsid/sentence_similarity_hindi')
embeddings = model.encode(sentences)
print(embeddings)
