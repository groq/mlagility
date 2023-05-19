# labels: test_group::monthly author::melll-uff name::bertweetbr downloads::249 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
import torch
from transformers import AutoModel, AutoTokenizer 

model = AutoModel.from_pretrained('melll-uff/bertweetbr')
tokenizer = AutoTokenizer.from_pretrained('melll-uff/bertweetbr', normalization=False)

# INPUT TWEETS ALREADY NORMALIZED!
inputs = [
    "Procuro um amor , que seja bom pra mim ... vou procurar , eu vou até o fim :nota_musical:",
    "Que jogo ontem @USER :mãos_juntas:",
    "Demojizer para Python é :polegar_para_cima: e está disponível em HTTPURL"]

encoded_inputs = tokenizer(inputs, return_tensors="pt", padding=True)

with torch.no_grad():
    last_hidden_states = model(**encoded_inputs)

# CLS Token of last hidden states. Shape: (number of input sentences, hidden sizeof the model)
last_hidden_states[0][:,0,:]

tensor([[-0.1430, -0.1325,  0.1595,  ..., -0.0802, -0.0153, -0.1358],
        [-0.0108,  0.1415,  0.0695,  ...,  0.1420,  0.1153, -0.0176],
        [-0.1854,  0.1866,  0.3163,  ..., -0.2117,  0.2123, -0.1907]])
