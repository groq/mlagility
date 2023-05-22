# labels: test_group::monthly author::henryk name::bert-base-multilingual-cased-finetuned-dutch-squad2 downloads::655 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="henryk/bert-base-multilingual-cased-finetuned-dutch-squad2",
    tokenizer="henryk/bert-base-multilingual-cased-finetuned-dutch-squad2"
)

qa_pipeline({
    'context': "Amsterdam is de hoofdstad en de dichtstbevolkte stad van Nederland.",
    'question': "Wat is de hoofdstad van Nederland?"})
