# labels: test_group::monthly author::cmarkea name::distilcamembert-base-nli downloads::425 license::mit task::Natural_Language_Processing sub_task::Zero-Shot_Classification
from transformers import pipeline

classifier = pipeline(
    task='zero-shot-classification',
    model="cmarkea/distilcamembert-base-nli",
    tokenizer="cmarkea/distilcamembert-base-nli"
)
result = classifier (
    sequences="Le style très cinéphile de Quentin Tarantino "
    "se reconnaît entre autres par sa narration postmoderne "
    "et non linéaire, ses dialogues travaillés souvent "
    "émaillés de références à la culture populaire, et ses "
    "scènes hautement esthétiques mais d'une violence "
    "extrême, inspirées de films d'exploitation, d'arts "
    "martiaux ou de western spaghetti.",
    candidate_labels="cinéma, technologie, littérature, politique",
    hypothesis_template="Ce texte parle de {}."
)

result
{"labels": ["cinéma",
            "littérature",
            "technologie",
            "politique"],
 "scores": [0.7164115309715271,
            0.12878799438476562,
            0.1092301607131958,
            0.0455702543258667]}
