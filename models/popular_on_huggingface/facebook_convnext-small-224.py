# labels: test_group::monthly,daily author::facebook name::convnext-small-224 downloads::1,084 license::apache-2.0 task::Computer_Vision sub_task::Image_Classification
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

feature_extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-small-224")
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-small-224")

inputs = feature_extractor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label]),
