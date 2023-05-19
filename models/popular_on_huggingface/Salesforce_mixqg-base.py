# labels: test_group::monthly author::Salesforce name::mixqg-base downloads::920 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import pipeline

nlp = pipeline("text2text-generation", model='Salesforce/mixqg-base', tokenizer='Salesforce/mixqg-base')
    
CONTEXT = "In the late 17th century, Robert Boyle proved that air is necessary for combustion."
ANSWER = "Robert Boyle"

def format_inputs(context: str, answer: str):
    return f"{answer} \\n {context}"

text = format_inputs(CONTEXT, ANSWER)

nlp(text)
# should output [{'generated_text': 'Who proved that air is necessary for combustion?'}]
