# labels: test_group::monthly author::mrm8488 name::bert-medium-finetuned-squadv2 downloads::26,836 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-small-finetuned-squadv2",
    tokenizer="mrm8488/bert-small-finetuned-squadv2"
)

qa_pipeline({
    'context': "Manuel Romero has been working hardly in the repository hugginface/transformers lately",
    'question': "Who has been working hard for hugginface/transformers lately?"

})

# Output:
