# labels: test_group::monthly author::ybelkada name::japanese-roberta-question-answering task::Natural_Language_Processing downloads::337 license::cc-by-sa-3.0
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
question = 'アレクサンダー・グラハム・ベルは、どこで生まれたの?'
context = 'アレクサンダー・グラハム・ベルは、スコットランド生まれの科学者、発明家、工学者である。世界初の>実用的電話の発明で知られている。'
model = AutoModelForQuestionAnswering.from_pretrained(
    'ybelkada/japanese-roberta-question-answering')
tokenizer = AutoTokenizer.from_pretrained(
    'ybelkada/japanese-roberta-question-answering')
inputs = tokenizer(
    question, context, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]
outputs = model(**inputs)
answer_start_scores = outputs.start_logits
answer_end_scores = outputs.end_logits
# Get the most likely beginning of answer with the argmax of the score.
answer_start = torch.argmax(answer_start_scores)
# Get the most likely end of answer with the argmax of the score.
# 1 is added to `answer_end` because the index pointed by score is inclusive.
answer_end = torch.argmax(answer_end_scores) + 1
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
# answer = 'スコットランド'
