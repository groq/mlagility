# labels: test_group::monthly author::TristanBehrens name::js-fakes-4bars downloads::199 task::Natural_Language_Processing sub_task::Text_Generation
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TristanBehrens/js-fakes-4bars")
model = AutoModelForCausalLM.from_pretrained("TristanBehrens/js-fakes-4bars")

input_ids = tokenizer.encode("PIECE_START", return_tensors="pt")
print(input_ids)

generated_ids = model.generate(input_ids, max_length=500)
generated_sequence = tokenizer.decode(generated_ids[0])
print(generated_sequence)
