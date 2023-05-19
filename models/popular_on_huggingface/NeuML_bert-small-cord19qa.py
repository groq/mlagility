# labels: test_group::monthly author::NeuML name::bert-small-cord19qa downloads::491 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import pipeline

qa = pipeline(
    "question-answering",
    model="NeuML/bert-small-cord19qa",
    tokenizer="NeuML/bert-small-cord19qa"
)

qa({
    "question": "What is the median incubation period?",
    "context": "The incubation period is around 5 days (range: 4-7 days) with a maximum of 12-13 day"
})

qa({
    "question": "What is the incubation period range?",
    "context": "The incubation period is around 5 days (range: 4-7 days) with a maximum of 12-13 day"
})

qa({
    "question": "What type of surfaces does it persist?",
    "context": "The virus can survive on surfaces for up to 72 hours such as plastic and stainless steel ."
})
