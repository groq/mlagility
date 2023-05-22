# labels: test_group::monthly author::uer name::albert-base-chinese-cluecorpussmall downloads::320,714 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import BertTokenizer, AlbertForMaskedLM, FillMaskPipeline
tokenizer = BertTokenizer.from_pretrained("uer/albert-base-chinese-cluecorpussmall")
model = AlbertForMaskedLM.from_pretrained("uer/albert-base-chinese-cluecorpussmall")
unmasker = FillMaskPipeline(model, tokenizer)   
unmasker("中国的首都是[MASK]京。")

