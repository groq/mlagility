# labels: test_group::monthly author::readerbench name::jurBERT-base task::Natural_Language_Processing downloads::1,296
# tensorflow
from transformers import AutoModel, AutoTokenizer, TFAutoModel
tokenizer = AutoTokenizer.from_pretrained("readerbench/jurBERT-base")
model = TFAutoModel.from_pretrained("readerbench/jurBERT-base")
inputs = tokenizer("exemplu de propoziție", return_tensors="tf")
outputs = model(inputs)


# pytorch
from transformers import AutoModel, AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("readerbench/jurBERT-base")
model = AutoModel.from_pretrained("readerbench/jurBERT-base")
inputs = tokenizer("exemplu de propoziție", return_tensors="pt")
outputs = model(**inputs)
