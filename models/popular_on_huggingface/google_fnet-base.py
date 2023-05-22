# labels: test_group::monthly,daily author::google name::fnet-base task::Natural_Language_Processing downloads::178,925 license::apache-2.0
from transformers import FNetForMaskedLM, FNetTokenizer, pipeline
tokenizer = FNetTokenizer.from_pretrained("google/fnet-base")
model = FNetForMaskedLM.from_pretrained("google/fnet-base")
unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
unmasker("Hello I'm a [MASK] model.")


