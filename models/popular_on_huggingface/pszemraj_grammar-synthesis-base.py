# labels: test_group::monthly author::pszemraj name::grammar-synthesis-base downloads::819 license::cc-by-nc-sa-4.0 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import pipeline
corrector = pipeline(
              'text2text-generation',
              'pszemraj/grammar-synthesis-base',
              )
raw_text = 'i can has cheezburger'
results = corrector(raw_text)
print(results)
