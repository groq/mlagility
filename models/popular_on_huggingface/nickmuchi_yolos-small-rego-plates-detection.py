# labels: test_group::monthly author::nickmuchi name::yolos-small-rego-plates-detection downloads::541 license::apache-2.0 task::Computer_Vision sub_task::Object_Detection
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import requests

url = 'https://drive.google.com/uc?id=1p9wJIqRz3W50e2f_A0D8ftla8hoXz4T5'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = YolosFeatureExtractor.from_pretrained('nickmuchi/yolos-small-rego-plates-detection')
model = YolosForObjectDetection.from_pretrained('nickmuchi/yolos-small-rego-plates-detection')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

# model predicts bounding boxes and corresponding face mask detection classes
logits = outputs.logits
bboxes = outputs.pred_boxes
