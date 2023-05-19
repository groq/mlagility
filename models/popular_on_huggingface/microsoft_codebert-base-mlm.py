# labels: test_group::monthly,daily author::microsoft name::codebert-base-mlm downloads::273,375 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline

model = RobertaForMaskedLM.from_pretrained('microsoft/codebert-base-mlm')
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')

code_example = "if (x is not None) <mask> (x>1)"
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

outputs = fill_mask(code_example)
print(outputs)
