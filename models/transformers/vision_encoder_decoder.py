# labels: test_group::mlagility name::vision_encoder_decoder author::huggingface_pytorch task::Computer_Vision
from mlagility.parser import parse
import transformers
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, height, max_seq_length, width = parse(
    ["batch_size", "height", "max_seq_length", "width"]
)

# Model and input configurations
config = transformers.VisionEncoderDecoderConfig.from_encoder_decoder_configs(
    transformers.ViTConfig(), transformers.BertConfig()
)
model = transformers.VisionEncoderDecoderModel(config)
inputs = {
    "pixel_values": torch.ones(
        batch_size, num_channels, height, width, dtype=torch.float
    ),
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
}


# Call model
model(**inputs)
