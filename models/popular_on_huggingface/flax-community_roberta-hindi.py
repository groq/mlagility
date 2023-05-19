# labels: test_group::monthly author::flax-community name::roberta-hindi downloads::373 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='flax-community/roberta-hindi')
unmasker("हम आपके सुखद <mask> की कामना करते हैं")

