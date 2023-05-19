# labels: test_group::monthly author::Rostlab name::prot_bert downloads::332,524 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import BertForMaskedLM, BertTokenizer, pipeline
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
unmasker('D L I P T S S K L V V [MASK] D T S L Q V K K A F F A L V T')


