# labels: test_group::monthly author::MilaNLProc name::feel-it-italian-emotion downloads::15,195 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
classifier = pipeline("text-classification",model='MilaNLProc/feel-it-italian-emotion',top_k=2)
prediction = classifier("Oggi sono proprio contento!")
print(prediction)
