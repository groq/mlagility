# labels: test_group::monthly author::aiknowyou name::aiky-sentence-bertino downloads::615 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["Questa è una frase di esempio", "Ogni frase viene convertita"]

model = SentenceTransformer('aiknowyou/aiky-sentence-bertino')
embeddings = model.encode(sentences)
print(embeddings)
