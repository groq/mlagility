# labels: test_group::monthly author::razent name::spbert-mlm-wso-base downloads::595 task::Natural_Language_Processing sub_task::Question_Answering
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('razent/spbert-mlm-wso-base')
model = AutoModel.from_pretrained("razent/spbert-mlm-wso-base")
text = "select * where brack_open var_a var_b var_c sep_dot brack_close"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
