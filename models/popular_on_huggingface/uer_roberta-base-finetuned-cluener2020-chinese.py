# labels: test_group::monthly author::uer name::roberta-base-finetuned-cluener2020-chinese downloads::5,409 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import AutoModelForTokenClassification,AutoTokenizer,pipeline
model = AutoModelForTokenClassification.from_pretrained('uer/roberta-base-finetuned-cluener2020-chinese')
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-cluener2020-chinese')
ner = pipeline('ner', model=model, tokenizer=tokenizer)
ner("江苏警方通报特斯拉冲进店铺")

