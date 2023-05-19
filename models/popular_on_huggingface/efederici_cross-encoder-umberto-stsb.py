# labels: test_group::monthly author::efederici name::cross-encoder-umberto-stsb downloads::247 task::Natural_Language_Processing sub_task::Text_Classification
from sentence_transformers import CrossEncoder
model = CrossEncoder('efederici/cross-encoder-umberto-stsb')
scores = model.predict([('Sentence 1', 'Sentence 2'), ('Sentence 3', 'Sentence 4')])
