# labels: name::swinv2_cr_small_224 author::timm task::Computer_Vision
import torch
import timm
from mlagility.parser import parse

# Parsing command-line arguments
batch_size = parse(["batch_size"])

# Creating model and set it to evaluation mode
model = timm.create_model("swinv2_cr_small_224", pretrained = False)
model.eval()

# Creating inputs
input_size = model.default_cfg["input_size"]
batched_input_size = tuple(batch_size) + input_size
inputs = torch.rand(batched_input_size)

# Calling model
model(inputs)
