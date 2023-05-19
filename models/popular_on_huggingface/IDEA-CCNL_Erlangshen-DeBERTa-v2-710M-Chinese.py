# labels: test_group::monthly author::IDEA-CCNL name::Erlangshen-DeBERTa-v2-710M-Chinese downloads::527 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import AutoModelForMaskedLM, AutoTokenizer, FillMaskPipeline
import torch

tokenizer=AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese', use_fast=False)
model=AutoModelForMaskedLM.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese')
text = '生活的真谛是[MASK]。'
fillmask_pipe = FillMaskPipeline(model, tokenizer, device=-1)
print(fillmask_pipe(text, top_k=10))
