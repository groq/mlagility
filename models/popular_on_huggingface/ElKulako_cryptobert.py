# labels: test_group::monthly author::ElKulako name::cryptobert downloads::740 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding = 'max_length')
# post_1 & post_3 = bullish, post_2 = bearish
post_1 = " see y'all tomorrow and can't wait to see ada in the morning, i wonder what price it is going to be at. 😎🐂🤠💯😴, bitcoin is looking good go for it and flash by that 45k. "
post_2 = "  alright racers, it’s a race to the bottom! good luck today and remember there are no losers (minus those who invested in currency nobody really uses) take your marks... are you ready? go!!" 
post_3 = " i'm never selling. the whole market can bottom out. i'll continue to hold this dumpster fire until the day i die if i need to." 
df_posts = [post_1, post_2, post_3]
preds = pipe(df_posts)
print(preds)

