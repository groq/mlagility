# labels: name::llama_7b_cache_layer author::transformers task::Natural_Language_Processing
from mlagility_models.llm_layer.llama_layer_prototype import call_llama_layer

call_llama_layer(params="7B", use_cache=True)
