# labels: test_group::monthly author::SkolkovoInstitute name::russian_toxicity_classifier downloads::4,336 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import BertTokenizer, BertForSequenceClassification

# load tokenizer and model weights
tokenizer = BertTokenizer.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')
model = BertForSequenceClassification.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')

# prepare the input
batch = tokenizer.encode('ты супер', return_tensors='pt')

# inference
model(batch)
