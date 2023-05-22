# labels: test_group::hot_at_groq,monthly,daily author::CompVis name::stable-diffusion-v1-4 downloads::933,179 license::creativeml-openrail-m task::Multimodal sub_task::Text-to-Image
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cpu"


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, guidance_scale=7.5).images[0]

image.save("astronaut_rides_horse.png")
