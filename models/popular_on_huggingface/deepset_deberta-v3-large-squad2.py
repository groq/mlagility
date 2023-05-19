# labels: test_group::monthly author::deepset name::deberta-v3-large-squad2 downloads::20,376 license::cc-by-4.0 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/deberta-v3-large-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}
res = nlp(QA_input)

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
