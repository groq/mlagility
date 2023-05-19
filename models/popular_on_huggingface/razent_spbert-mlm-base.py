# labels: test_group::monthly author::razent name::spbert-mlm-base downloads::310 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('razent/spbert-mlm-base')
model = AutoModel.from_pretrained("razent/spbert-mlm-base")
text = "select * where brack_open var_a var_b var_c sep_dot brack_close"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
