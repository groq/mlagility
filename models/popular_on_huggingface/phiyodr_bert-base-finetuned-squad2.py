# labels: test_group::monthly author::phiyodr name::bert-base-finetuned-squad2 downloads::2,083 task::Natural_Language_Processing sub_task::Question_Answering
from transformers.pipelines import pipeline

model_name = "phiyodr/bert-base-finetuned-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
inputs = {
    'question': 'What discipline did Winkelmann create?',
    'context': 'Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. "The prophet and founding hero of modern archaeology", Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art. '
}
nlp(inputs)
