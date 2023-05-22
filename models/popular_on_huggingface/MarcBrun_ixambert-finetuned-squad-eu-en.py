# labels: test_group::monthly author::MarcBrun name::ixambert-finetuned-squad-eu-en downloads::886 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "MarcBrun/ixambert-finetuned-squad-eu-en"

# To get predictions
context = "Florence Nightingale, known for being the founder of modern nursing, was born in Florence, Italy, in 1820"
question = "When was Florence Nightingale born?"
qa = pipeline("question-answering", model=model_name, tokenizer=model_name)
pred = qa(question=question,context=context)

# To load the model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
