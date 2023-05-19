# labels: test_group::monthly author::IlyaGusev name::rubert_telegram_headlines downloads::227 license::apache-2.0 task::Natural_Language_Processing sub_task::Summarization
from transformers import AutoTokenizer, EncoderDecoderModel

model_name = "IlyaGusev/rubert_telegram_headlines"
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, do_basic_tokenize=False, strip_accents=False)
model = EncoderDecoderModel.from_pretrained(model_name)

article_text = "..."

input_ids = tokenizer(
    [article_text],
    add_special_tokens=True,
    max_length=256,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    max_length=64,
    no_repeat_ngram_size=3,
    num_beams=10,
    top_p=0.95
)[0]

headline = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(headline)
