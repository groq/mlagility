# labels: test_group::monthly author::KoichiYasuoka name::deberta-base-japanese-aozora-ud-head downloads::358 license::cc-by-sa-4.0 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,QuestionAnsweringPipeline
tokenizer=AutoTokenizer.from_pretrained("KoichiYasuoka/deberta-base-japanese-aozora-ud-head")
model=AutoModelForQuestionAnswering.from_pretrained("KoichiYasuoka/deberta-base-japanese-aozora-ud-head")
qap=QuestionAnsweringPipeline(tokenizer=tokenizer,model=model,align_to_words=False)
print(qap(question="国語",context="全学年にわたって小学校の国語の教科書に挿し絵>が用いられている"))
