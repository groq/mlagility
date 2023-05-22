# labels: test_group::monthly author::flexudy name::t5-base-multi-sentence-doctor downloads::206,333 task::Natural_Language_Processing sub_task::Text2Text_Generation

from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("flexudy/t5-base-multi-sentence-doctor")

model = AutoModelWithLMHead.from_pretrained("flexudy/t5-base-multi-sentence-doctor")

input_text = "repair_sentence: m a medical doct context: {That is my job I a}{or I save lives} </s>"

input_ids = tokenizer.encode(input_text, return_tensors="pt")

outputs = model.generate(input_ids, max_length=32, num_beams=1)

sentence = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

assert sentence == "I am a medical doctor."
