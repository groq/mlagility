# labels: test_group::monthly author::taeminlee name::kogpt2 downloads::1,006 task::Natural_Language_Processing sub_task::Text_Generation
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

model = GPT2LMHeadModel.from_pretrained("taeminlee/kogpt2")
tokenizer = PreTrainedTokenizerFast.from_pretrained("taeminlee/kogpt2")

input_ids = tokenizer.encode("안녕", add_special_tokens=False, return_tensors="pt")
output_sequences = model.generate(input_ids=input_ids, do_sample=True, max_length=100, num_return_sequences=3)
for generated_sequence in output_sequences:
    generated_sequence = generated_sequence.tolist()
    print("GENERATED SEQUENCE : {0}".format(tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)))
