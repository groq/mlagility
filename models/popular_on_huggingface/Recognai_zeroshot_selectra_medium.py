# labels: test_group::monthly author::Recognai name::zeroshot_selectra_medium downloads::716 license::apache-2.0 task::Natural_Language_Processing sub_task::Zero-Shot_Classification
from transformers import pipeline
classifier = pipeline("zero-shot-classification", 
                       model="Recognai/zeroshot_selectra_medium")

classifier(
    "El autor se perfila, a los 50 años de su muerte, como uno de los grandes de su siglo",
    candidate_labels=["cultura", "sociedad", "economia", "salud", "deportes"],
    hypothesis_template="Este ejemplo es {}."
)
"""Output
{'sequence': 'El autor se perfila, a los 50 años de su muerte, como uno de los grandes de su siglo',
 'labels': ['sociedad', 'cultura', 'economia', 'salud', 'deportes'],
 'scores': [0.6450043320655823,
  0.16710571944713593,
  0.08507631719112396,
  0.0759836807847023,
  0.026829993352293968]}
"""
