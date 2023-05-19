# labels: test_group::monthly author::batterydata name::batterybert-cased-squad-v1 downloads::202 license::apache-2.0 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "batterydata/batterybert-cased-squad-v1"
# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'What is the electrolyte?',
    'context': 'The typical non-aqueous electrolyte for commercial Li-ion cells is a solution of LiPF6 in linear and cyclic carbonates.'
}
res = nlp(QA_input)
# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
