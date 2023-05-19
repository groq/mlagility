# labels: test_group::monthly author::nkoh01 name::MSRoberta downloads::705 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline,AutoModelForMaskedLM,AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nkoh01/MSRoberta")
model = AutoModelForMaskedLM.from_pretrained("nkoh01/MSRoberta")

unmasker = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)
unmasker("Hello, it is a <mask> to meet you.")

[{'score': 0.9508683085441589,
  'sequence': 'hello, it is a pleasure to meet you.',
  'token': 10483,
  'token_str': ' pleasure'},
 {'score': 0.015089659951627254,
  'sequence': 'hello, it is a privilege to meet you.',
  'token': 9951,
  'token_str': ' privilege'},
 {'score': 0.013942377641797066,
  'sequence': 'hello, it is a joy to meet you.',
  'token': 5823,
  'token_str': ' joy'},
 {'score': 0.006964420434087515,
  'sequence': 'hello, it is a delight to meet you.',
  'token': 13213,
  'token_str': ' delight'},
 {'score': 0.0024567877408117056,
  'sequence': 'hello, it is a honour to meet you.',
  'token': 6671,
  'token_str': ' honour'}]
