# labels: test_group::monthly author::jhgan name::ko-sbert-multitask downloads::206 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
sentences = ["안녕하세요?", "한국어 문장 임베딩을 위한 버트 모델입니다."]

model = SentenceTransformer('jhgan/ko-sbert-multitask')
embeddings = model.encode(sentences)
print(embeddings)
