# labels: test_group::monthly author::cmarkea name::distilcamembert-base-sentiment downloads::2,818 license::mit task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline

analyzer = pipeline(
    task='text-classification',
    model="cmarkea/distilcamembert-base-sentiment",
    tokenizer="cmarkea/distilcamembert-base-sentiment"
)
result = analyzer(
    "J'aime me promener en forêt même si ça me donne mal aux pieds.",
    return_all_scores=True
)

result
[{'label': '1 star',
  'score': 0.047529436647892},
 {'label': '2 stars',
  'score': 0.14150355756282806},
 {'label': '3 stars',
  'score': 0.3586442470550537},
 {'label': '4 stars',
  'score': 0.3181498646736145},
 {'label': '5 stars',
  'score': 0.13417290151119232}]
