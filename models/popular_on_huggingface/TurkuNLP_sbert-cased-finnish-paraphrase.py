# labels: test_group::monthly author::TurkuNLP name::sbert-cased-finnish-paraphrase downloads::205 task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["Tämä on esimerkkilause.", "Tämä on toinen lause."]

model = SentenceTransformer('TurkuNLP/sbert-cased-finnish-paraphrase')
embeddings = model.encode(sentences)
print(embeddings)
