# labels: test_group::monthly author::maxpe name::twitter-roberta-base-jun2022_sem_eval_2018_task_1 downloads::520 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline

pipe = pipeline("text-classification",model="maxpe/twitter-roberta-base-jun2022_sem_eval_2018_task_1")

pipe("I couldn't see any seafood for a year after I went to that restaurant that they send all the tourists to!",top_k=11)
