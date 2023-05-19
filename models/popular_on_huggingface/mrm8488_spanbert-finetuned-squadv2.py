# labels: test_group::monthly author::mrm8488 name::spanbert-finetuned-squadv2 downloads::1,981 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/spanbert-finetuned-squadv2",
    tokenizer="mrm8488/spanbert-finetuned-squadv2"
)

qa_pipeline({
    'context': "Manuel Romero has been working hardly in the repository hugginface/transformers lately",
    'question': "Who has been working hard for hugginface/transformers lately?"

})

# Output: {'answer': 'Manuel Romero','end': 13,'score': 6.836378586818937e-09, 'start': 0}
