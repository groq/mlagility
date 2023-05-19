# labels: test_group::monthly author::Sahajtomar name::German_Zeroshot downloads::20,169 task::Natural_Language_Processing sub_task::Zero-Shot_Classification
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="Sahajtomar/German_Zeroshot")
sequence = "Letzte Woche gab es einen Selbstmord in einer nahe gelegenen kolonie"
candidate_labels = ["Verbrechen","Tragödie","Stehlen"]
hypothesis_template = "In deisem geht es um {}."    ## Since monolingual model,its sensitive to hypothesis template. This can be experimented

classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template)
"""{'labels': ['Tragödie', 'Verbrechen', 'Stehlen'],
 'scores': [0.8328856854438782, 0.10494536352157593, 0.06316883927583696],
 'sequence': 'Letzte Woche gab es einen Selbstmord in einer nahe gelegenen Kolonie'}"""
