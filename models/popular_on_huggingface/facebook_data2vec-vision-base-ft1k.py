# labels: test_group::monthly,daily author::facebook name::data2vec-vision-base-ft1k downloads::896 license::apache-2.0 task::Computer_Vision sub_task::Image_Classification
from transformers import BeitFeatureExtractor, Data2VecVisionForImageClassification
from PIL import Image
import requests
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = BeitFeatureExtractor.from_pretrained('facebook/data2vec-vision-base-ft1k')
model = Data2VecVisionForImageClassification.from_pretrained('facebook/data2vec-vision-base-ft1k')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
