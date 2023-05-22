# labels: test_group::monthly author::Xuhui name::ToxDect-roberta-large downloads::642 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
classifier = pipeline("text-classification",model='Xuhui/ToxDect-roberta-large', return_all_scores=True)
prediction = classifier("You are f**king stupid!", )
print(prediction)

"""
Output:
[[{'label': 'LABEL_0', 'score': 0.002632011892274022}, {'label': 'LABEL_1', 'score': 0.9973680377006531}]]
"""
