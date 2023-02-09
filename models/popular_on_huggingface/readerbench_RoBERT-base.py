# labels: test_group::monthly author::readerbench name::RoBERT-base task::unknown downloads::265
# tensorflow
from transformers import AutoModel, AutoTokenizer, TFAutoModel
tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-base")
model = TFAutoModel.from_pretrained("readerbench/RoBERT-base")
inputs = tokenizer("exemplu de propoziție", return_tensors="tf")
outputs = model(inputs)

# pytorch
from transformers import AutoModel, AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-base")
model = AutoModel.from_pretrained("readerbench/RoBERT-base")
inputs = tokenizer("exemplu de propoziție", return_tensors="pt")
outputs = model(**inputs)
