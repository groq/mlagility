# labels: test_group::monthly author::clips name::mfaq downloads::428,140 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer

question = "<Q>How many models can I host on HuggingFace?"
answer_1 = "<A>All plans come with unlimited private models and datasets."
answer_2 = "<A>AutoNLP is an automatic way to train and deploy state-of-the-art NLP models, seamlessly integrated with the Hugging Face ecosystem."
answer_3 = "<A>Based on how much training data and model variants are created, we send you a compute cost and payment link - as low as $10 per job."

model = SentenceTransformer('clips/mfaq')
embeddings = model.encode([question, answer_1, answer_3, answer_3])
print(embeddings)
