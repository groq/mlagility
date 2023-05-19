# labels: test_group::mlagility name::safety_clipvision author::diffusers task::Generative_AI
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# Use Stable Diffusion to instantiate the Safety ClipVision model, then return the model

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    safety_checker=StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    ),
)

model = pipe.safety_checker.vision_model
inputs = {"pixel_values": torch.ones([1, 3, 224, 224], dtype=torch.float)}

# Call model
model(**inputs)
