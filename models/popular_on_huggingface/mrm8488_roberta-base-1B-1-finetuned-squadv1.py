# labels: test_group::monthly author::mrm8488 name::roberta-base-1B-1-finetuned-squadv1 downloads::185 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import pipeline

QnA_pipeline = pipeline('question-answering', model='mrm8488/roberta-base-1B-1-finetuned-squadv1')

QnA_pipeline({
    'context': 'A new strain of flu that has the potential to become a pandemic has been identified in China by scientists.',
    'question': 'What has been discovered by scientists from China ?'
})
# Output:

{'answer': 'A new strain of flu', 'end': 19, 'score': 0.04702283976040074, 'start': 0}
