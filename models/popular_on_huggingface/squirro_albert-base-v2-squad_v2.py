# labels: test_group::monthly author::squirro name::albert-base-v2-squad_v2 downloads::199 license::apache-2.0 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline
model = AutoModelForQuestionAnswering.from_pretrained("squirro/albert-base-v2-squad_v2")
tokenizer = AutoTokenizer.from_pretrained("squirro/albert-base-v2-squad_v2")
qa_model = QuestionAnsweringPipeline(model, tokenizer)
qa_model(
   question="What's your name?",
   context="My name is Clara and I live in Berkeley.",
   handle_impossible_answer=True  # important!
)

