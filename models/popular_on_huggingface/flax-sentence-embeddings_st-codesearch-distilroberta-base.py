# labels: test_group::monthly author::flax-sentence-embeddings name::st-codesearch-distilroberta-base downloads::853 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer, util


#This list the defines the different programm codes
code = ["""def sort_list(x):
   return sorted(x)""",
"""def count_above_threshold(elements, threshold=0):
    counter = 0
    for e in elements:
        if e > threshold:
            counter += 1
    return counter""",
"""def find_min_max(elements):
    min_ele = 99999
    max_ele = -99999
    for e in elements:
        if e < min_ele:
            min_ele = e
        if e > max_ele:
            max_ele = e
    return min_ele, max_ele"""]
    

model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")

# Encode our code into the vector space
code_emb = model.encode(code, convert_to_tensor=True)

# Interactive demo: Enter queries, and the method returns the best function from the 
# 3 functions we defined
while True:
    query = input("Query: ")
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, code_emb)[0]
    top_hit = hits[0]

    print("Cossim: {:.2f}".format(top_hit['score']))
    print(code[top_hit['corpus_id']])
    print("\n\n")
