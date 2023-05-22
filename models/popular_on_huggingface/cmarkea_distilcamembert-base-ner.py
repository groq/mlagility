# labels: test_group::monthly author::cmarkea name::distilcamembert-base-ner downloads::6,491 license::mit task::Natural_Language_Processing sub_task::Token_Classification
from transformers import pipeline

ner = pipeline(
    task='ner',
    model="cmarkea/distilcamembert-base-ner",
    tokenizer="cmarkea/distilcamembert-base-ner",
    aggregation_strategy="simple"
)
result = ner(
    "Le Crédit Mutuel Arkéa est une banque Française, elle comprend le CMB "
    "qui est une banque située en Bretagne et le CMSO qui est une banque "
    "qui se situe principalement en Aquitaine. C'est sous la présidence de "
    "Louis Lichou, dans les années 1980 que différentes filiales sont créées "
    "au sein du CMB et forment les principales filiales du groupe qui "
    "existent encore aujourd'hui (Federal Finance, Suravenir, Financo, etc.)."
)

result
[{'entity_group': 'ORG',
  'score': 0.9974479,
  'word': 'Crédit Mutuel Arkéa',
  'start': 3,
  'end': 22},
 {'entity_group': 'LOC',
  'score': 0.9000358,
  'word': 'Française',
  'start': 38,
  'end': 47},
 {'entity_group': 'ORG',
  'score': 0.9788757,
  'word': 'CMB',
  'start': 66,
  'end': 69},
 {'entity_group': 'LOC',
  'score': 0.99919766,
  'word': 'Bretagne',
  'start': 99,
  'end': 107},
 {'entity_group': 'ORG',
  'score': 0.9594884,
  'word': 'CMSO',
  'start': 114,
  'end': 118},
 {'entity_group': 'LOC',
  'score': 0.99935514,
  'word': 'Aquitaine',
  'start': 169,
  'end': 178},
 {'entity_group': 'PER',
  'score': 0.99911094,
  'word': 'Louis Lichou',
  'start': 208,
  'end': 220},
 {'entity_group': 'ORG',
  'score': 0.96226394,
  'word': 'CMB',
  'start': 291,
  'end': 294},
 {'entity_group': 'ORG',
  'score': 0.9983959,
  'word': 'Federal Finance',
  'start': 374,
  'end': 389},
 {'entity_group': 'ORG',
  'score': 0.9984454,
  'word': 'Suravenir',
  'start': 391,
  'end': 400},
 {'entity_group': 'ORG',
  'score': 0.9985084,
  'word': 'Financo',
  'start': 402,
  'end': 409}]
