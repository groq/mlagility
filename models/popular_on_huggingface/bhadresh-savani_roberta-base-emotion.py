# labels: test_group::monthly author::bhadresh-savani name::roberta-base-emotion downloads::480 license::apache-2.0 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
classifier = pipeline("text-classification",model='bhadresh-savani/roberta-base-emotion', return_all_scores=True)
prediction = classifier("I love using transformers. The best part is wide range of support and its easy to use", )
print(prediction)

"""
Output:
[[
{'label': 'sadness', 'score': 0.002281982684507966}, 
{'label': 'joy', 'score': 0.9726489186286926}, 
{'label': 'love', 'score': 0.021365027874708176}, 
{'label': 'anger', 'score': 0.0026395076420158148}, 
{'label': 'fear', 'score': 0.0007162453257478774}, 
{'label': 'surprise', 'score': 0.0003483477921690792}
]]
"""
