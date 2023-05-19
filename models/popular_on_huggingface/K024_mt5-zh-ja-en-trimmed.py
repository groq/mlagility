# labels: test_group::monthly author::K024 name::mt5-zh-ja-en-trimmed downloads::36,651 license::cc-by-nc-sa-4.0 task::Natural_Language_Processing sub_task::Translation
from transformers import (
  T5Tokenizer,
  MT5ForConditionalGeneration,
  Text2TextGenerationPipeline,
)

path = "K024/mt5-zh-ja-en-trimmed"
pipe = Text2TextGenerationPipeline(
  model=MT5ForConditionalGeneration.from_pretrained(path),
  tokenizer=T5Tokenizer.from_pretrained(path),
)

sentence = "ja2zh: 吾輩は猫である。名前はまだ無い。"
res = pipe(sentence, max_length=100, num_beams=4)
res[0]['generated_text']
