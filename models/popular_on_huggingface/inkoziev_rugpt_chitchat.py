# labels: test_group::monthly author::inkoziev name::rugpt_chitchat downloads::434 license::unlicense task::Natural_Language_Processing sub_task::Text_Generation
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "inkoziev/rugpt_chitchat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
model.eval()

# На вход модели подаем последние 2-3 реплики диалога. Каждая реплика на отдельной строке, начинается с символа "-"
input_text = """<s>- Привет! Что делаешь?
- Привет :) В такси еду
-"""

encoded_prompt = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt").to(device)

output_sequences = model.generate(input_ids=encoded_prompt, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)

text = tokenizer.decode(output_sequences[0].tolist(), clean_up_tokenization_spaces=True)[len(input_text)+1:]
text = text[: text.find('</s>')]
print(text)
