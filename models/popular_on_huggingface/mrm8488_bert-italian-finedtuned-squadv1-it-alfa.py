# labels: test_group::monthly author::mrm8488 name::bert-italian-finedtuned-squadv1-it-alfa downloads::3,249 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import pipeline

nlp_qa = pipeline(
    'question-answering',
    model='mrm8488/bert-italian-finedtuned-squadv1-it-alfa',
    tokenizer='mrm8488/bert-italian-finedtuned-squadv1-it-alfa'
)

nlp_qa(
    {
        'question': 'Per quale lingua stai lavorando?',
        'context': 'Manuel Romero è colaborando attivamente con HF / trasformatori per il trader del poder de las últimas ' +
       'técnicas di procesamiento de lenguaje natural al idioma español'
    }
)

# Output: {'answer': 'español', 'end': 174, 'score': 0.9925341537498156, 'start': 168}
