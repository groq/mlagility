# labels: test_group::monthly author::uer name::roberta-base-finetuned-jd-binary-chinese downloads::452 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoModelForSequenceClassification,AutoTokenizer,pipeline
model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-chinanews-chinese')
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-chinanews-chinese')
text_classification = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
text_classification("北京上个月召开了两会")

