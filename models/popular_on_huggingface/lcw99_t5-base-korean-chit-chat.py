# labels: test_group::monthly author::lcw99 name::t5-base-korean-chit-chat downloads::243 task::Natural_Language_Processing sub_task::Text2Text_Generation

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MT5ForConditionalGeneration
from transformers import AutoTokenizer, T5TokenizerFast
import nltk
nltk.download('punkt')


model_dir = f"lcw99/t5-base-korean-chit-chat"

max_input_length = 1024

text = """
A: 쇼핑하러 갈까? B: 응 좋아. A: 언제 갈까? B:
"""

inputs = [text]

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=3, do_sample=True, min_length=20, max_length=500, num_return_sequences=3)
for i in range(3):
    #print(output[i])
    print("---", i)
    decoded_output = tokenizer.decode(output[i], skip_special_tokens=True)
    predicted_title = nltk.sent_tokenize(decoded_output)
    #print(decoded_output)
    print(predicted_title)

import torch

chat_history = []
# Let's chat for 5 lines
for step in range(100):
    print("")
    user_input = input(">> User: ")
    chat_history.append("A: " + user_input)
    while len(chat_history) > 5:
        chat_history.pop(0)
    hist = ""
    for chat in chat_history:
        hist += "\n" + chat
    hist += "\nB: "
    new_user_input_ids = tokenizer.encode(hist, return_tensors='pt')

    bot_input_ids = new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(
        bot_input_ids, max_length=200,
        pad_token_id=tokenizer.eos_token_id,  
        do_sample=True, 
        #top_k=100, 
        #top_p=0.7,
        #temperature = 0.1
    )

    bot_text = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True).replace("#@이름#", "OOO")
    bot_text = bot_text.replace("\n", " / ")
    chat_history.append("B: " + bot_text)
    
    # pretty print last ouput tokens from bot
    print("Bot: {}".format(bot_text))    
