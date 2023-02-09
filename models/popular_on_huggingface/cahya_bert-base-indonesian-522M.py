# labels: test_group::monthly author::cahya name::bert-base-indonesian-522M downloads::1,826 license::mit task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='cahya/bert-base-indonesian-522M')
unmasker("Ibu ku sedang bekerja [MASK] supermarket")


