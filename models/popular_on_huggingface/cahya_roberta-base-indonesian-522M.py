# labels: test_group::monthly author::cahya name::roberta-base-indonesian-522M downloads::372 license::mit task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='cahya/roberta-base-indonesian-522M')
unmasker("Ibu ku sedang bekerja <mask> supermarket")

