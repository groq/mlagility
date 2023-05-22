# labels: test_group::monthly author::recobo name::agriculture-bert-uncased downloads::369 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
fill_mask = pipeline(
    "fill-mask",
    model="recobo/agriculture-bert-uncased",
    tokenizer="recobo/agriculture-bert-uncased"
)
fill_mask("[MASK] is the practice of cultivating plants and livestock.")
