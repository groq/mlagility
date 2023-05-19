# labels: test_group::monthly author::whaleloops name::phrase-bert downloads::395 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
phrase_list = [ 'play an active role', 'participate actively', 'active lifestyle']

model = SentenceTransformer('whaleloops/phrase-bert')
phrase_embs = model.encode( phrase_list )
[p1, p2, p3] = phrase_embs
