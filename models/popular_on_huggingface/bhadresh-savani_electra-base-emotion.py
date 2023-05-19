# labels: test_group::monthly author::bhadresh-savani name::electra-base-emotion downloads::348 license::apache-2.0 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
classifier = pipeline("text-classification",model='bhadresh-savani/electra-base-emotion', return_all_scores=True)
prediction = classifier("I love using transformers. The best part is wide range of support and its easy to use", )
print(prediction)

"""
Output:
[[
{'label': 'sadness', 'score': 0.0006792712374590337}, 
{'label': 'joy', 'score': 0.9959300756454468}, 
{'label': 'love', 'score': 0.0009452480007894337}, 
{'label': 'anger', 'score': 0.0018055217806249857}, 
{'label': 'fear', 'score': 0.00041110432357527316}, 
{'label': 'surprise', 'score': 0.0002288572577526793}
]]
"""
