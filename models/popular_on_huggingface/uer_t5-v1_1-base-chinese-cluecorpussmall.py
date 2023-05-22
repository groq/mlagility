# labels: test_group::monthly author::uer name::t5-v1_1-base-chinese-cluecorpussmall downloads::1,673 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import BertTokenizer, MT5ForConditionalGeneration, Text2TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("uer/t5-v1_1-small-chinese-cluecorpussmall")
model = MT5ForConditionalGeneration.from_pretrained("uer/t5-v1_1-small-chinese-cluecorpussmall")
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)  
text2text_generator("中国的首都是extra0京", max_length=50, do_sample=False)

