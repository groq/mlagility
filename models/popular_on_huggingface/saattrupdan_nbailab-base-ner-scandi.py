# labels: test_group::monthly author::saattrupdan name::nbailab-base-ner-scandi downloads::4,067 license::mit task::Natural_Language_Processing sub_task::Token_Classification
from transformers import pipeline
import pandas as pd
ner = pipeline(task='ner', 
               model='saattrupdan/nbailab-base-ner-scandi', 
               aggregation_strategy='first')
result = ner('Borghild kjøper seg inn i Bunnpris')
pd.DataFrame.from_records(result)

