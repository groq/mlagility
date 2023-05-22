# labels: test_group::monthly author::paust name::pko-t5-base downloads::624 license::cc-by-4.0 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import T5TokenizerFast, T5ForConditionalGeneration

tokenizer = T5TokenizerFast.from_pretrained('paust/pko-t5-base')
model = T5ForConditionalGeneration.from_pretrained('paust/pko-t5-base')

input_ids = tokenizer(["qa question: 당신의 이름은 무엇인가요?"]).input_ids
labels = tokenizer(["T5 입니다."]).input_ids
outputs = model(input_ids=input_ids, labels=labels)

print(f"loss={outputs.loss} logits={outputs.logits}")
