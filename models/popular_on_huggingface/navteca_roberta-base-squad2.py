# labels: test_group::monthly author::navteca name::roberta-base-squad2 downloads::906 license::mit task::Natural_Language_Processing sub_task::Question_Answering
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Load model & tokenizer
roberta_model = AutoModelForQuestionAnswering.from_pretrained('navteca/roberta-base-squad2')
roberta_tokenizer = AutoTokenizer.from_pretrained('navteca/roberta-base-squad2')

# Get predictions
nlp = pipeline('question-answering', model=roberta_model, tokenizer=roberta_tokenizer)

result = nlp({
    'question': 'How many people live in Berlin?',
    'context': 'Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.'
})

print(result)

#{
#  "answer": "3,520,031"
#  "end": 36,
#  "score": 0.96186668,
#  "start": 27,
#}
