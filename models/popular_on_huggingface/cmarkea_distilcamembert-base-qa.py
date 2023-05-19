# labels: test_group::monthly author::cmarkea name::distilcamembert-base-qa downloads::2,400 license::cc-by-nc-sa-3.0 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import pipeline

qa_engine = pipeline(
    "question-answering",
    model="cmarkea/distilcamembert-base-qa",
    tokenizer="cmarkea/distilcamembert-base-qa"
)

result = qa_engine(
    context="David Fincher, né le 28 août 1962 à Denver (Colorado), "
    "est un réalisateur et producteur américain. Il est principalement "
    "connu pour avoir réalisé les films Seven, Fight Club, L'Étrange "
    "Histoire de Benjamin Button, The Social Network et Gone Girl qui "
    "lui ont valu diverses récompenses et nominations aux Oscars du "
    "cinéma ou aux Golden Globes. Réputé pour son perfectionnisme, il "
    "peut tourner un très grand nombre de prises de ses plans et "
    "séquences afin d'obtenir le rendu visuel qu'il désire. Il a "
    "également développé et produit les séries télévisées House of "
    "Cards (pour laquelle il remporte l'Emmy Award de la meilleure "
    "réalisation pour une série dramatique en 2013) et Mindhunter, "
    "diffusées sur Netflix.",
    question="Quel est le métier de David Fincher ?"
)

result
{'score': 0.7981914281845093,
 'start': 61,
 'end': 98,
 'answer': ' réalisateur et producteur américain.'}
