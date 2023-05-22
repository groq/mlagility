# labels: test_group::monthly author::salesken name::paraphrase_diversity_ranker downloads::681 license::apache-2.0 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification  
import torch
import pandas as pd
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("salesken/paraphrase_diversity_ranker")
model = AutoModelForSequenceClassification.from_pretrained("salesken/paraphrase_diversity_ranker")

input_query = ["tough challenges make you stronger."]
paraphrases =  [
        "tough problems make you stronger",
        "tough problems will make you stronger",
        "tough challenges make you stronger",
        "tough challenges will make you a stronger person",
        "tough challenges will make you stronger",
        "tough tasks make you stronger",
        "the tough task makes you stronger",
        "tough stuff makes you stronger",
        "if tough times make you stronger",
        "the tough part makes you stronger",
        "tough issues strengthens you",
        "tough shit makes you stronger",
        "tough tasks force you to be stronger",
        "tough challenge is making you stronger",
        "tough problems make you have more strength"]
para_pairs=list(pd.MultiIndex.from_product([input_query, paraphrases]))


features = tokenizer(para_pairs,  padding=True, truncation=True, return_tensors="pt")
model.eval()
with torch.no_grad():
    scores = model(**features).logits
    label_mapping = ['surface_level_variation', 'semantic_variation']
    labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]

sorted_diverse_paraphrases= np.array(para_pairs)[scores[:,1].sort(descending=True).indices].tolist()
print(sorted_diverse_paraphrases)


# to identify the type of paraphrase (surface-level variation or semantic variation) 
print("Paraphrase type detection=====", list(zip(para_pairs, labels)))
