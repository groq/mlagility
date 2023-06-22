"""
This file defines a generic function for instantiating and calling a layer of the LLaMA model.
There are other scripts in this directory (models/llm_layer) that call this function with a
variety of arguments. We implement this way to avoid duplicating the same source code across
many llama_*b_*.py scripts.
"""

from mlagility.parser import parse
import transformers
import torch


def call_llama_layer(params="7B", use_cache=False):

    torch.manual_seed(0)

    # Parsing command-line arguments
    batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])

    # Configuration
    configs = {
        "7B": transformers.LlamaConfig(
            hidden_size=4096,
            num_attention_heads=32,
            intermediate_size=4096 * 4,
            use_cache=use_cache,
        ),
        "13B": transformers.LlamaConfig(
            hidden_size=5120,
            num_attention_heads=40,
            intermediate_size=5120 * 4,
            use_cache=use_cache,
        ),
        "33B": transformers.LlamaConfig(
            hidden_size=6656,
            num_attention_heads=52,
            intermediate_size=6656 * 4,
            use_cache=use_cache,
        ),
        "65B": transformers.LlamaConfig(
            hidden_size=8192,
            num_attention_heads=64,
            intermediate_size=8192 * 4,
            use_cache=use_cache,
        ),
    }
    config = configs[params]

    # Model
    model = transformers.models.llama.modeling_llama.LlamaDecoderLayer(config)

    # Make sure the user's sequence length fits within the model's maximum
    assert max_seq_length <= config.max_position_embeddings

    if use_cache:
        inputs = {
            "hidden_states": torch.ones(
                batch_size,
                1,
                config.hidden_size,
                dtype=torch.float,
            ),
            "attention_mask": torch.ones(
                batch_size,
                1,
                1,
                max_seq_length,
                dtype=torch.float,
            ),
            "position_ids": [[0]],
            "past_key_value": (
                torch.ones(
                    batch_size,
                    config.num_attention_heads,
                    max_seq_length - 1,
                    model.self_attn.head_dim,
                    dtype=torch.float,
                ),
                torch.ones(
                    batch_size,
                    config.num_attention_heads,
                    max_seq_length - 1,
                    model.self_attn.head_dim,
                    dtype=torch.float,
                ),
            ),
        }
    else:
        inputs = {
            "hidden_states": torch.ones(
                batch_size, max_seq_length, config.hidden_size, dtype=torch.float
            ),
            "attention_mask": torch.ones(
                batch_size, 1, max_seq_length, max_seq_length, dtype=torch.float
            ),
        }

    # Call layer
    model(**inputs)
