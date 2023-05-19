# labels: test_group::monthly author::flax-sentence-embeddings name::all_datasets_v4_MiniLM-L6 downloads::8,563 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
text = "Replace me by any text you'd like."
text_embbedding = model.encode(text)
# array([-0.01559514,  0.04046123,  0.1317083 ,  0.00085931,  0.04585106,
#        -0.05607086,  0.0138078 ,  0.03569756,  0.01420381,  0.04266302 ...],
#        dtype=float32)
