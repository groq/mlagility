# labels: test_group::monthly,daily author::google name::vit-base-patch32-224-in21k downloads::3,348 license::apache-2.0 task::Multimodal sub_task::Feature_Extraction
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch32-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
