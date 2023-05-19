# labels: test_group::monthly author::gerulata name::slovakbert downloads::1,284 license::mit task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='gerulata/slovakbert')
unmasker("Deti sa <mask> na ihrisku.")

[{'sequence': 'Deti sa hrali na ihrisku.',
  'score': 0.6355380415916443,
  'token': 5949,
  'token_str': ' hrali'},
 {'sequence': 'Deti sa hrajú na ihrisku.',
  'score': 0.14731724560260773,
  'token': 9081,
  'token_str': ' hrajú'},
 {'sequence': 'Deti sa zahrali na ihrisku.',
  'score': 0.05016357824206352,
  'token': 32553,
  'token_str': ' zahrali'},
 {'sequence': 'Deti sa stretli na ihrisku.',
  'score': 0.041727423667907715,
  'token': 5964,
  'token_str': ' stretli'},
 {'sequence': 'Deti sa učia na ihrisku.',
  'score': 0.01886524073779583,
  'token': 18099,
  'token_str': ' učia'}]
