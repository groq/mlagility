# labels: test_group::monthly author::edumunozsala name::vit_base-224-in21k-ft-cifar100 downloads::242 license::apache-2.0 task::Computer_Vision sub_task::Image_Classification
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('edumunozsala/vit_base-224-in21k-ft-cifar100')
inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
