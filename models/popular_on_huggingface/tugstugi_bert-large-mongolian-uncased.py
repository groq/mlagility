# labels: test_group::monthly author::tugstugi name::bert-large-mongolian-uncased downloads::648 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('tugstugi/bert-large-mongolian-uncased', use_fast=False)
model = AutoModelForMaskedLM.from_pretrained('tugstugi/bert-large-mongolian-uncased')

## declare task ##
pipe = pipeline(task="fill-mask", model=model, tokenizer=tokenizer)

## example ##
input_  = 'Монгол улсын [MASK] Улаанбаатар хотоос ярьж байна.'

output_ = pipe(input_)
for i in range(len(output_)):
    print(output_[i])

## output ##
# {'sequence': 'монгол улсын нийслэл улаанбаатар хотоос ярьж байна.', 'score': 0.7867621183395386, 'token': 849, 'token_str': 'нийслэл'}
# {'sequence': 'монгол улсын ерөнхийлөгч улаанбаатар хотоос ярьж байна.', 'score': 0.14303277432918549, 'token': 244, 'token_str': 'ерөнхийлөгч'}
# {'sequence': 'монгол улсын ерөнхийлөгчийг улаанбаатар хотоос ярьж байна.', 'score': 0.011642335914075375, 'token': 8373, 'token_str': 'ерөнхийлөгчийг'}
# {'sequence': 'монгол улсын иргэд улаанбаатар хотоос ярьж байна.', 'score': 0.006592822726815939, 'token': 247, 'token_str': 'иргэд'}
# {'sequence': 'монгол улсын нийслэлийг улаанбаатар хотоос ярьж байна.', 'score': 0.006165097933262587, 'token': 15501, 'token_str': 'нийслэлийг'}
