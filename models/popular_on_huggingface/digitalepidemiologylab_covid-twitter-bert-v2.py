# labels: test_group::monthly author::digitalepidemiologylab name::covid-twitter-bert-v2 task::Natural_Language_Processing downloads::8,270 license::mit
from transformers import pipeline
import json

pipe = pipeline(task='fill-mask', model='digitalepidemiologylab/covid-twitter-bert-v2')
out = pipe(f"In places with a lot of people, it's a good idea to wear a {pipe.tokenizer.mask_token}")
print(json.dumps(out, indent=4))
[
    {
        "sequence": "[CLS] in places with a lot of people, it's a good idea to wear a mask [SEP]",
        "score": 0.9998226761817932,
        "token": 7308,
        "token_str": "mask"
    },
    ...
]
