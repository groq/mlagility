# labels: test_group::monthly author::fran-martinez name::scibert_scivocab_cased_ner_jnlpba downloads::10,836 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import pipeline

text = "Mouse thymus was used as a source of glucocorticoid receptor from normal CS lymphocytes."

nlp_ner = pipeline("ner",
                   model='fran-martinez/scibert_scivocab_cased_ner_jnlpba',
                   tokenizer='fran-martinez/scibert_scivocab_cased_ner_jnlpba')

nlp_ner(text)

"""
Output:
---------------------------
[
{'word': 'glucocorticoid', 
'score': 0.9894881248474121, 
'entity': 'B-protein'}, 
 
{'word': 'receptor', 
'score': 0.989505410194397, 
'entity': 'I-protein'}, 

{'word': 'normal', 
'score': 0.7680378556251526, 
'entity': 'B-cell_type'}, 

{'word': 'cs', 
'score': 0.5176806449890137, 
'entity': 'I-cell_type'}, 

{'word': 'lymphocytes', 
'score': 0.9898491501808167, 
'entity': 'I-cell_type'}
]
"""
