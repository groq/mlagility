# labels: test_group::monthly,daily author::facebook name::convnext-xlarge-224-22k downloads::950 license::apache-2.0 task::Computer_Vision sub_task::Image_Classification
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

feature_extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-xlarge-224-22k")
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-xlarge-224-22k")

inputs = feature_extractor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 22k ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label]),
