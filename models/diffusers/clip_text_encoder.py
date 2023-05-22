# labels: test_group::mlagility name::clip_text_encoder author::diffusers task::MultiModal
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# Use Stable Diffusion to instantiate the CLIP text embedding model, then return the model

from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")

model = pipe.text_encoder
inputs = {"input_ids": torch.ones([1, 77], dtype=torch.int)}

# Call model
model(**inputs)
