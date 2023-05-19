# labels: test_group::monthly author::KoichiYasuoka name::deberta-base-thai-ud-head downloads::418 license::apache-2.0 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,QuestionAnsweringPipeline
tokenizer=AutoTokenizer.from_pretrained("KoichiYasuoka/deberta-base-thai-ud-head")
model=AutoModelForQuestionAnswering.from_pretrained("KoichiYasuoka/deberta-base-thai-ud-head")
qap=QuestionAnsweringPipeline(tokenizer=tokenizer,model=model,align_to_words=False)
print(qap(question="กว่า",context="หลายหัวดีกว่าหัวเดียว"))
