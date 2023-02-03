# labels: test_group::mlagility name::safety_clipvision author::diffusers
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# Use Stable Diffusion to instantiate the Safety ClipVision model, then return the model

from diffusers import StableDiffusionPipeline
import torch


pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token="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",
)

model = pipe.safety_checker.vision_model
inputs = {"pixel_values": torch.ones([1, 3, 224, 224], dtype=torch.float)}


# Call model
model(**inputs)
