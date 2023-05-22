# labels: test_group::monthly author::tennessejoyce name::titlewave-t5-base downloads::6,394 license::cc-by-4.0 task::Natural_Language_Processing sub_task::Summarization
from transformers import pipeline
classifier = pipeline('summarization', model='tennessejoyce/titlewave-t5-base')
body = """"Example question body."""
classifier(body)


