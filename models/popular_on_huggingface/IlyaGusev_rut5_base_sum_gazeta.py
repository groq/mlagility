# labels: test_group::monthly author::IlyaGusev name::rut5_base_sum_gazeta downloads::469 license::apache-2.0 task::Natural_Language_Processing sub_task::Summarization
from transformers import AutoTokenizer, T5ForConditionalGeneration

model_name = "IlyaGusev/rut5_base_sum_gazeta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

article_text = "..."

input_ids = tokenizer(
    [article_text],
    max_length=600,
    add_special_tokens=True,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    no_repeat_ngram_size=4
)[0]

summary = tokenizer.decode(output_ids, skip_special_tokens=True)
print(summary)
