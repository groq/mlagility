# labels: test_group::monthly author::SkolkovoInstitute name::roberta_toxicity_classifier downloads::15,220 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# load tokenizer and model weights
tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

# prepare the input
batch = tokenizer.encode('you are amazing', return_tensors='pt')

# inference
model(batch)
