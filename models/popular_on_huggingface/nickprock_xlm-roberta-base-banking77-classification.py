# labels: test_group::monthly author::nickprock name::xlm-roberta-base-banking77-classification downloads::210 license::mit task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
pipe = pipeline("text-classification", model="nickprock/xlm-roberta-base-banking77-classification")
pipe("Non riesco a pagare con la carta di credito")
