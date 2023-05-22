# labels: test_group::monthly author::chkla name::parlbert-topic-german downloads::35,916 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline

pipeline_classification_topics = pipeline("text-classification", model="chkla/parlbert-topic-german", tokenizer="bert-base-german-cased", return_all_scores=False)

text = "Sachgebiet Ausschließliche Gesetzgebungskompetenz des Bundes über die Zusammenarbeit des Bundes und der Länder zum Schutze der freiheitlichen demokratischen Grundordnung, des Bestandes und der Sicherheit des Bundes oder eines Landes Wir fragen die Bundesregierung"

pipeline_classification_topics(text) # Government
