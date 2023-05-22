# labels: test_group::monthly author::AmazonScience name::qanlu downloads::374 license::cc-by-4.0 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
  
tokenizer = AutoTokenizer.from_pretrained("AmazonScience/qanlu", use_auth_token=True)

model = AutoModelForQuestionAnswering.from_pretrained("AmazonScience/qanlu", use_auth_token=True)

qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

qa_input = {
  'context': 'Yes. No. I want a cheap flight to Boston.',
  'question': 'What is the destination?'
}

answer = qa_pipeline(qa_input)
