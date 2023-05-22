# labels: test_group::monthly author::mrm8488 name::spanbert-large-finetuned-squadv2 task::Natural_Language_Processing downloads::389
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/spanbert-large-finetuned-squadv2",
    tokenizer="SpanBERT/spanbert-large-cased"
)

qa_pipeline({
    'context': "Manuel Romero has been working very hard in the repository hugginface/transformers lately",
    'question': "How has been working Manuel Romero lately?"

})
# Output: {'answer': 'very hard', 'end': 40, 'score': 0.9052708846768347, 'start': 31}
