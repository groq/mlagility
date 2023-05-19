# labels: test_group::monthly author::kiri-ai name::distiluse-base-multilingual-cased-et downloads::196 task::Multimodal sub_task::Feature_Extraction
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('kiri-ai/distiluse-base-multilingual-cased-et')
sentences = ['Here is a sample sentence','Another sample sentence']
embeddings = model.encode(sentences)

print("Sentence embeddings:")
print(embeddings)
