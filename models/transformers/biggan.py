# labels: test_group::mlagility name::biggan author::transformers task::Generative_AI
"""https://github.com/huggingface/pytorch-pretrained-BigGAN/blob/master/README.md"""
from mlagility.parser import parse
import torch

from pytorch_pretrained_biggan import BigGAN


# Parsing command-line arguments
batch_size = parse(["batch_size"])
noise_dim = 128
class_vector_dim = 1000
truncation_dim = 1

# Model and input configurations
model = BigGAN.from_pretrained("biggan-deep-256")

inputs = {
    "z": torch.ones(batch_size, noise_dim, dtype=torch.float),
    "class_label": torch.ones(batch_size, class_vector_dim, dtype=torch.float),
    "truncation": torch.ones(truncation_dim, dtype=torch.float),
}


# Call model
model(**inputs)
