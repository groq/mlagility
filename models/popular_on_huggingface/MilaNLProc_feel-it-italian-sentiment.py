# labels: test_group::monthly author::MilaNLProc name::feel-it-italian-sentiment downloads::16,971 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
classifier = pipeline("text-classification",model='MilaNLProc/feel-it-italian-sentiment',top_k=2)
prediction = classifier("Oggi sono proprio contento!")
print(prediction)
