# labels: test_group::monthly author::cahya name::distilbert-base-indonesian downloads::353 license::mit task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='cahya/distilbert-base-indonesian')
unmasker("Ayahku sedang bekerja di sawah untuk [MASK] padi")


