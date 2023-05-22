# labels: test_group::mlagility name::dcgan author::torch_hub task::Generative_AI
"""https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_dcgan/"""

from mlagility.parser import parse
import torch

# Parsing command-line arguments
batch_size = parse(["batch_size"])
noise_dim = 128

# Model and input configurations
model = torch.hub.load("facebookresearch/pytorch_GAN_zoo:hub", "DCGAN", pretrained=True)

inputs = {"z": torch.ones([batch_size, noise_dim], dtype=torch.float)}


# Call model
model(**inputs)
