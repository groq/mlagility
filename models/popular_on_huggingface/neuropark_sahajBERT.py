# labels: test_group::monthly author::neuropark name::sahajBERT downloads::341 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask

from transformers import AlbertForMaskedLM, FillMaskPipeline, PreTrainedTokenizerFast

# Initialize tokenizer

tokenizer = PreTrainedTokenizerFast.from_pretrained("neuropark/sahajBERT")

# Initialize model

model = AlbertForMaskedLM.from_pretrained("neuropark/sahajBERT")

# Initialize pipeline

pipeline = FillMaskPipeline(tokenizer=tokenizer, model=model)

raw_text = "ধন্যবাদ। আপনার সাথে কথা [MASK] ভালো লাগলো" # Change me

pipeline(raw_text)
