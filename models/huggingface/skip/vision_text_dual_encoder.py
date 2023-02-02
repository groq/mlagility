# labels: name::vision_text_dual_encoder author::skip
import mlagility
import transformers
import torch


# Parsing command-line arguments
batch_size, height, max_seq_length, num_channels, width = mlagility.parse(
    ["batch_size", "height", "max_seq_length", "num_channels", "width"]
)

# Model and input configurations
config = transformers.VisionTextDualEncoderConfig.from_vision_text_configs(
    transformers.ViTConfig(), transformers.BertConfig(), projection_dim=512
)
model = transformers.VisionTextDualEncoderModel(config)
inputs = {
    "pixel_values": torch.ones(
        batch_size, num_channels, height, width, dtype=torch.float
    ),
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
}


# Call model
model(**inputs)
