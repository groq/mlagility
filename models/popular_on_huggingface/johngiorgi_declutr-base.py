# labels: test_group::monthly author::johngiorgi name::declutr-base downloads::894 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("johngiorgi/declutr-base")

# Prepare some text to embed
texts = [
    "A smiling costumed woman is holding an umbrella.",
    "A happy woman in a fairy costume holds an umbrella.",
]

# Embed the text
embeddings = model.encode(texts)

# Compute a semantic similarity via the cosine distance
semantic_sim = 1 - cosine(embeddings[0], embeddings[1])
