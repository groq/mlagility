# labels: test_group::mlagility name::vae_decoder author::diffusers task::Generative_AI
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# Use Stable Diffusion to instantiate the VAE decoder model, then return the VAE

from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")

model = pipe.vae.decoder
inputs = {"z": torch.ones([1, 4, 64, 64], dtype=torch.float)}

# Call model
model(**inputs)
