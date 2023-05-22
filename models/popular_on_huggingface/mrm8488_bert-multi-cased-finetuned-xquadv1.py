# labels: test_group::monthly author::mrm8488 name::bert-multi-cased-finetuned-xquadv1 downloads::930 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import pipeline

from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-multi-cased-finetuned-xquadv1",
    tokenizer="mrm8488/bert-multi-cased-finetuned-xquadv1"
)


# context: Coronavirus is seeding panic in the West because it expands so fast.

# question: Where is seeding panic Coronavirus?
qa_pipeline({
    'context': "कोरोनावायरस पश्चिम में आतंक बो रहा है क्योंकि यह इतनी तेजी से फैलता है।",
    'question': "कोरोनावायरस घबराहट कहां है?"
    
})
# output: {'answer': 'पश्चिम', 'end': 18, 'score': 0.7037217439689059, 'start': 12}

qa_pipeline({
    'context': "Manuel Romero has been working hardly in the repository hugginface/transformers lately",
    'question': "Who has been working hard for hugginface/transformers lately?"
    
})
# output: {'answer': 'Manuel Romero', 'end': 13, 'score': 0.7254485993702389, 'start': 0}

qa_pipeline({
    'context': "Manuel Romero a travaillé à peine dans le référentiel hugginface / transformers ces derniers temps",
    'question': "Pour quel référentiel a travaillé Manuel Romero récemment?"
    
})
#output: {'answer': 'hugginface / transformers', 'end': 79, 'score': 0.6482061613915384, 'start': 54}
