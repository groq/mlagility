# labels: test_group::monthly,daily author::facebook name::vit-mae-base sub_task::unknown downloads::11,994 license::apache-2.0
from transformers import AutoFeatureExtractor, ViTMAEForPreTraining
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/vit-mae-base')
model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
loss = outputs.loss
mask = outputs.mask
ids_restore = outputs.ids_restore
