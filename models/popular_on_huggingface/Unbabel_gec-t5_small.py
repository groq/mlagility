# labels: test_group::monthly author::Unbabel name::gec-t5_small downloads::389 license::apache-2.0 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("Unbabel/gec-t5_small")
tokenizer = T5Tokenizer.from_pretrained('t5-small')

sentence = "I like to swimming"
tokenized_sentence = tokenizer('gec: ' + sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
corrected_sentence = tokenizer.decode(
    model.generate(
        input_ids = tokenized_sentence.input_ids,
        attention_mask = tokenized_sentence.attention_mask, 
        max_length=128,
        num_beams=5,
        early_stopping=True,
    )[0],
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=True
)
print(corrected_sentence) # -> I like swimming.
