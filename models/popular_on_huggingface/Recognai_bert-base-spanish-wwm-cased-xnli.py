# labels: test_group::monthly author::Recognai name::bert-base-spanish-wwm-cased-xnli downloads::9,931 license::mit task::Natural_Language_Processing sub_task::Zero-Shot_Classification
from transformers import pipeline
classifier = pipeline("zero-shot-classification", 
                       model="Recognai/bert-base-spanish-wwm-cased-xnli")

classifier(
    "El autor se perfila, a los 50 años de su muerte, como uno de los grandes de su siglo",
    candidate_labels=["cultura", "sociedad", "economia", "salud", "deportes"],
    hypothesis_template="Este ejemplo es {}."
)
"""output
{'sequence': 'El autor se perfila, a los 50 años de su muerte, como uno de los grandes de su siglo',
 'labels': ['cultura', 'sociedad', 'economia', 'salud', 'deportes'],
 'scores': [0.38897448778152466,
  0.22997373342514038,
  0.1658431738615036,
  0.1205764189362526,
  0.09463217109441757]}
"""
