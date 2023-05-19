# labels: test_group::monthly,daily author::google name::electra-base-generator downloads::30,181 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="google/electra-base-generator",
    tokenizer="google/electra-base-generator"
)

print(
    fill_mask(f"HuggingFace is creating a {fill_mask.tokenizer.mask_token} that the community uses to solve NLP tasks.")
)
