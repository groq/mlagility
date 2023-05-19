# labels: test_group::monthly author::ans name::vaccinating-covid-tweets downloads::3,089 license::apache-2.0 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
pipe = pipeline("sentiment-analysis", model = "ans/vaccinating-covid-tweets")
seq = "Vaccines to prevent SARS-CoV-2 infection are considered the most promising approach for curbing the pandemic."
pipe(seq)
