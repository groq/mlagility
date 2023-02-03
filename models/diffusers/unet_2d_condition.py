# labels: test_group::mlagility name::unet_2d_condition author::diffusers
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# Use Stable Diffusion to instantiate the unet, then return the unet

from diffusers import StableDiffusionPipeline
import torch


pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token="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",
)

model = pipe.unet

latent_model_shape = [2, 4, 64, 64]
text_embeddings_shape = [2, 77, 768]

inputs = {
    "sample": torch.ones(latent_model_shape, dtype=torch.float),
    "timestep": 1,
    "encoder_hidden_states": torch.ones(text_embeddings_shape, dtype=torch.float),
}


# Call model
model(**inputs)
