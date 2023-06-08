# labels: name::llama_65b_layer author::transformers task::Generative_AI
from mlagility_models.llm_layer.llama_layer_prototype import call_llama_layer

call_llama_layer(params="65B", use_cache=False)
