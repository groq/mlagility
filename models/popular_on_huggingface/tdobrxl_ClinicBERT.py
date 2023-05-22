# labels: test_group::monthly author::tdobrxl name::ClinicBERT downloads::434 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import RobertaModel, RobertaTokenizer
model = RobertaModel.from_pretrained("tdobrxl/ClinicBERT")
tokenizer = RobertaTokenizer.from_pretrained("tdobrxl/ClinicBERT")

text = "Randomized Study of Shark Cartilage in Patients With Breast Cancer."
last_hidden_state, pooler_output = model(tokenizer.encode(text, return_tensors="pt")).last_hidden_state, model(tokenizer.encode(text, return_tensors="pt")).pooler_output
