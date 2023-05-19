# labels: test_group::monthly author::dangvantuan name::sentence-camembert-large downloads::13,347 license::apache-2.0 task::Natural_Language_Processing sub_task::Sentence_Similarity
from sentence_transformers import SentenceTransformer
model =  SentenceTransformer("dangvantuan/sentence-camembert-large")

sentences = ["Un avion est en train de décoller.",
          "Un homme joue d'une grande flûte.",
          "Un homme étale du fromage râpé sur une pizza.",
          "Une personne jette un chat au plafond.",
          "Une personne est en train de plier un morceau de papier.",
          ]

embeddings = model.encode(sentences)
