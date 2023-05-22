# labels: test_group::monthly author::readerbench name::RoBERT-large task::Natural_Language_Processing downloads::368
# tensorflow
from transformers import AutoModel, AutoTokenizer, TFAutoModel
tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-large")
model = TFAutoModel.from_pretrained("readerbench/RoBERT-large")
inputs = tokenizer("exemplu de propoziție", return_tensors="tf")
outputs = model(inputs)

# pytorch
from transformers import AutoModel, AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-large")
model = AutoModel.from_pretrained("readerbench/RoBERT-large")
inputs = tokenizer("exemplu de propoziție", return_tensors="pt")
outputs = model(**inputs)
