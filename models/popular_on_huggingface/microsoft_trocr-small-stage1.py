# labels: test_group::monthly,daily author::microsoft name::trocr-small-stage1 task::MultiModal downloads::585
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import torch

# load image from the IAM database
url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-stage1')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-stage1')

# training
pixel_values = processor(image, return_tensors="pt").pixel_values  # Batch size 1
decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]])
outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
