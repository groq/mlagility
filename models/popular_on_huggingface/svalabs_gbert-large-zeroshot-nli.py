# labels: test_group::monthly author::svalabs name::gbert-large-zeroshot-nli downloads::1,288 task::Natural_Language_Processing sub_task::Zero-Shot_Classification

from transformers import pipeline

zershot_pipeline = pipeline("zero-shot-classification",
                             model="svalabs/gbert-large-zeroshot-nli")

sequence = "Ich habe ein Problem mit meinem Iphone das so schnell wie möglich gelöst werden muss" 
labels = ["Computer", "Handy", "Tablet", "dringend", "nicht dringend"] 
hypothesis_template = "In diesem Satz geht es um das Thema {}."    


zershot_pipeline(sequence, labels, hypothesis_template=hypothesis_template)
