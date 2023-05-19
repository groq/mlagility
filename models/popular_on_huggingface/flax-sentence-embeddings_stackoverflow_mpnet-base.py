# labels: test_group::monthly author::flax-sentence-embeddings name::stackoverflow_mpnet-base downloads::2,520 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('flax-sentence-embeddings/stackoverflow_mpnet-base')
text = "Replace me by any question / answer you'd like."
text_embbedding = model.encode(text)
# array([-0.01559514,  0.04046123,  0.1317083 ,  0.00085931,  0.04585106,
#        -0.05607086,  0.0138078 ,  0.03569756,  0.01420381,  0.04266302 ...],
#        dtype=float32)
