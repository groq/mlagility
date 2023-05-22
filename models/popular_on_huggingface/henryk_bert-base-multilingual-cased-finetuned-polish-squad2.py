# labels: test_group::monthly author::henryk name::bert-base-multilingual-cased-finetuned-polish-squad2 downloads::1,551 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="henryk/bert-base-multilingual-cased-finetuned-polish-squad2",
    tokenizer="henryk/bert-base-multilingual-cased-finetuned-polish-squad2"
)

qa_pipeline({
    'context': "Warszawa jest największym miastem w Polsce pod względem liczby ludności i powierzchni",
    'question': "Jakie jest największe miasto w Polsce?"})
