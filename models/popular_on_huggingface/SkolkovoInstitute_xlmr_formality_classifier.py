# labels: test_group::monthly author::SkolkovoInstitute name::xlmr_formality_classifier downloads::302 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification

# load tokenizer and model weights
tokenizer = XLMRobertaTokenizerFast.from_pretrained('SkolkovoInstitute/xlmr_formality_classifier')
model = XLMRobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/xlmr_formality_classifier')

# prepare the input
batch = tokenizer.encode('ты супер', return_tensors='pt')

# inference
model(batch)
