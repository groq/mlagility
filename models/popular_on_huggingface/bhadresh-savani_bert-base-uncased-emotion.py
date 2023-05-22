# labels: test_group::monthly author::bhadresh-savani name::bert-base-uncased-emotion downloads::3,209 license::apache-2.0 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
classifier = pipeline("text-classification",model='bhadresh-savani/bert-base-uncased-emotion', return_all_scores=True)
prediction = classifier("I love using transformers. The best part is wide range of support and its easy to use", )
print(prediction)

"""
output:
[[
{'label': 'sadness', 'score': 0.0005138228880241513}, 
{'label': 'joy', 'score': 0.9972520470619202}, 
{'label': 'love', 'score': 0.0007443308713845909}, 
{'label': 'anger', 'score': 0.0007404946954920888}, 
{'label': 'fear', 'score': 0.00032938539516180754}, 
{'label': 'surprise', 'score': 0.0004197491507511586}
]]
"""
