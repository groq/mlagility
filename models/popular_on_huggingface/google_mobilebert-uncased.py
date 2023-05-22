# labels: test_group::monthly,daily author::google name::mobilebert-uncased task::Natural_Language_Processing downloads::48,600 license::apache-2.0
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="google/mobilebert-uncased",
    tokenizer="google/mobilebert-uncased"
)

print(
    fill_mask(f"HuggingFace is creating a {fill_mask.tokenizer.mask_token} that the community uses to solve NLP tasks.")
)
