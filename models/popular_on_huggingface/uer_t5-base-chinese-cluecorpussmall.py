# labels: test_group::monthly author::uer name::t5-base-chinese-cluecorpussmall downloads::624 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import BertTokenizer, T5ForConditionalGeneration, Text2TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("uer/t5-small-chinese-cluecorpussmall")
model = T5ForConditionalGeneration.from_pretrained("uer/t5-small-chinese-cluecorpussmall")
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)  
text2text_generator("中国的首都是extra0京", max_length=50, do_sample=False)

