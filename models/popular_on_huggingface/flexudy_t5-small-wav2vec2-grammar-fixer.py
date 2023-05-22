# labels: test_group::monthly author::flexudy name::t5-small-wav2vec2-grammar-fixer task::Natural_Language_Processing downloads::486
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "flexudy/t5-small-wav2vec2-grammar-fixer"

tokenizer = T5Tokenizer.from_pretrained(model_name)

model = T5ForConditionalGeneration.from_pretrained(model_name)

sent = """GOING ALONG SLUSHY COUNTRY ROADS AND SPEAKING TO DAMP AUDIENCES IN DRAUGHTY SCHOOL ROOMS DAY AFTER DAY FOR A FORTNIGHT HE'LL HAVE TO PUT IN AN APPEARANCE AT SOME PLACE OF WORSHIP ON SUNDAY MORNING AND HE CAN COME TO US IMMEDIATELY AFTERWARDS"""

input_text = "fix: { " + sent + " } </s>"

input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True, add_special_tokens=True)

outputs = model.generate(
    input_ids=input_ids,
    max_length=256,
    num_beams=4,
    repetition_penalty=1.0,
    length_penalty=1.0,
    early_stopping=True
)

sentence = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(f"{sentence}")
