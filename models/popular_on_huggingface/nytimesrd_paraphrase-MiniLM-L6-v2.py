# labels: test_group::monthly author::nytimesrd name::paraphrase-MiniLM-L6-v2 downloads::181 license::apache-2.0 task::Multimodal sub_task::Feature_Extraction
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
