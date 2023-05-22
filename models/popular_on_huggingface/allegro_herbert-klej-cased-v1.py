# labels: test_group::monthly author::allegro name::herbert-klej-cased-v1 task::Natural_Language_Processing downloads::1,312
from transformers import XLMTokenizer, RobertaModel

tokenizer = XLMTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
model = RobertaModel.from_pretrained("allegro/herbert-klej-cased-v1")

encoded_input = tokenizer.encode("Kto ma lepszą sztukę, ma lepszy rząd – to jasne.", return_tensors='pt')
outputs = model(encoded_input)
