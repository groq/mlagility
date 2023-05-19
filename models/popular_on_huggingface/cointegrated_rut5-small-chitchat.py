# labels: test_group::monthly author::cointegrated name::rut5-small-chitchat downloads::188 license::mit task::Natural_Language_Processing sub_task::Text2Text_Generation
# !pip install transformers sentencepiece
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("cointegrated/rut5-small-chitchat")
model = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-small-chitchat")

text = 'Привет! Расскажи, как твои дела?'
inputs = tokenizer(text, return_tensors='pt')
with torch.no_grad():
    hypotheses = model.generate(
        **inputs, 
        do_sample=True, top_p=0.5, num_return_sequences=3, 
        repetition_penalty=2.5,
        max_length=32,
    )
for h in hypotheses:
    print(tokenizer.decode(h, skip_special_tokens=True))
# Как обычно.
# Сейчас - в порядке.
# Хорошо.
# Wall time: 363 ms 
