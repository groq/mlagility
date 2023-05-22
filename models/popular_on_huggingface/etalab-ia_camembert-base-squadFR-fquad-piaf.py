# labels: test_group::monthly author::etalab-ia name::camembert-base-squadFR-fquad-piaf downloads::6,559 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import pipeline

nlp = pipeline('question-answering', model='etalab-ia/camembert-base-squadFR-fquad-piaf', tokenizer='etalab-ia/camembert-base-squadFR-fquad-piaf')

nlp({
    'question': "Qui est Claude Monet?",
    'context': "Claude Monet, né le 14 novembre 1840 à Paris et mort le 5 décembre 1926 à Giverny, est un peintre français et l’un des fondateurs de l'impressionnisme."
})
