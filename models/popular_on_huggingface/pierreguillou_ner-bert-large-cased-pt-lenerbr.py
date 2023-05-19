# labels: test_group::monthly author::pierreguillou name::ner-bert-large-cased-pt-lenerbr downloads::449 task::Natural_Language_Processing sub_task::Token_Classification
# install pytorch: check https://pytorch.org/
# !pip install transformers 
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# parameters
model_name = "pierreguillou/ner-bert-large-cased-pt-lenerbr"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = "Acrescento que não há de se falar em violação do artigo 114, § 3º, da Constituição Federal, posto que referido dispositivo revela-se impertinente, tratando da possibilidade de ajuizamento de dissídio coletivo pelo Ministério Público do Trabalho nos casos de greve em atividade essencial."

# tokenization
inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt")
tokens = inputs.tokens()

# get predictions
outputs = model(**inputs).logits
predictions = torch.argmax(outputs, dim=2)

# print predictions
for token, prediction in zip(tokens, predictions[0].numpy()):
    print((token, model.config.id2label[prediction]))
