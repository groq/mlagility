# labels: test_group::monthly author::projecte-aina name::roberta-base-ca-v2 downloads::229 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline
from pprint import pprint
tokenizer_hf = AutoTokenizer.from_pretrained('projecte-aina/roberta-base-ca-v2')
model = AutoModelForMaskedLM.from_pretrained('projecte-aina/roberta-base-ca-v2')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)
text = f"Em dic <mask>."
res_hf = pipeline(text)
pprint([r['token_str'] for r in res_hf])
