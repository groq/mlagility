# labels: test_group::monthly,daily author::facebook name::flava-full task::MultiModal downloads::5,282 license::bsd-3-clause
from PIL import Image
import requests

from transformers import FlavaProcessor, FlavaModel

model = FlavaModel.from_pretrained("facebook/flava-full")
processor = FlavaProcessor.from_pretrained("facebook/flava-full")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
  text=["a photo of a cat", "a photo of a dog"], images=[image, image], return_tensors="pt", padding="max_length", max_length=77
)

outputs = model(**inputs)
image_embeddings = outputs.image_embeddings # Batch size X (Number of image patches + 1) x Hidden size => 2 X 197 X 768
text_embeddings = outputs.text_embeddings # Batch size X (Text sequence length + 1) X Hidden size => 2 X 77 X 768
multimodal_embeddings = outputs.multimodal_embeddings # Batch size X (Number of image patches + Text Sequence Length + 3) X Hidden size => 2 X 275 x 768
# Multimodal embeddings can be used for multimodal tasks such as VQA


## Pass only image
from transformers import FlavaFeatureExtractor

feature_extractor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
inputs = feature_extractor(images=[image, image], return_tensors="pt")
outputs = model(**inputs)
image_embeddings = outputs.image_embeddings

## Pass only image
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")
inputs = tokenizer(["a photo of a cat", "a photo of a dog"], return_tensors="pt", padding="max_length", max_length=77)
outputs = model(**inputs)
text_embeddings = outputs.text_embeddings
