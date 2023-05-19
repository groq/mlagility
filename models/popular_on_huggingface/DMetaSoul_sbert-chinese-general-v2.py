# labels: test_group::monthly author::DMetaSoul name::sbert-chinese-general-v2 downloads::349 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["我的儿子！他猛然间喊道，我的儿子在哪儿？", "我的儿子呢！他突然喊道，我的儿子在哪里？"]

model = SentenceTransformer('DMetaSoul/sbert-chinese-general-v2')
embeddings = model.encode(sentences)
print(embeddings)
