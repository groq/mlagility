# labels: test_group::monthly author::Visual-Attention-Network name::van-base downloads::621 license::apache-2.0 task::Computer_Vision sub_task::Image_Classification
from transformers import AutoFeatureExtractor, VanForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

feature_extractor = AutoFeatureExtractor.from_pretrained("Visual-Attention-Network/van-base")
model = VanForImageClassification.from_pretrained("Visual-Attention-Network/van-base")

inputs = feature_extractor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])

