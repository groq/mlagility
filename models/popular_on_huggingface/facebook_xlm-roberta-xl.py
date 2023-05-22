# labels: test_group::monthly,daily author::facebook name::xlm-roberta-xl downloads::958 license::mit task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='facebook/xlm-roberta-xl')
unmasker("Europe is a <mask> continent.")


