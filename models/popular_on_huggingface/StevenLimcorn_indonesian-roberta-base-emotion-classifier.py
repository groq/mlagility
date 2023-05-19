# labels: test_group::monthly author::StevenLimcorn name::indonesian-roberta-base-emotion-classifier downloads::251 license::mit task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
pretrained_name = "StevenLimcorn/indonesian-roberta-base-emotion-classifier"
nlp = pipeline(
    "sentiment-analysis",
    model=pretrained_name,
    tokenizer=pretrained_name
)
nlp("Hal-hal baik akan datang.")
