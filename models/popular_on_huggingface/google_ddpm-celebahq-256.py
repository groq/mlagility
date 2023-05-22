# labels: test_group::monthly,daily author::google name::ddpm-celebahq-256 downloads::1,827 license::apache-2.0 task::Computer_Vision sub_task::Unconditional_Image_Generation
# !pip install diffusers
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

model_id = "google/ddpm-celebahq-256"

# load model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

# run pipeline in inference (sample random noise and denoise)
image = ddpm()["sample"]


# save image
image[0].save("ddpm_generated_image.png")
