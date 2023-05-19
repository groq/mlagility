# labels: test_group::monthly author::deepset name::all-mpnet-base-v2-table downloads::7,007 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('deepset/all-mpnet-base-v2-table')
embeddings = model.encode(sentences)
print(embeddings)
