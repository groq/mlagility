# labels: test_group::mlagility name::vae_decoder author::diffusers
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# Use Stable Diffusion to instantiate the VAE decoder model, then return the VAE

from diffusers import StableDiffusionPipeline
import torch


pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token="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",
)

model = pipe.vae.decoder
inputs = {"z": torch.ones([1, 4, 64, 64], dtype=torch.float)}


# Call model
model(**inputs)
