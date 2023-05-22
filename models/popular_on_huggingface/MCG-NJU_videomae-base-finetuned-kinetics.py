# labels: test_group::monthly author::MCG-NJU name::videomae-base-finetuned-kinetics task::Computer_Vision downloads::1,449 license::cc-by-nc-4.0
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
import numpy as np
import torch

video = list(np.random.randn(16, 3, 224, 224))

feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

inputs = feature_extractor(video, return_tensors="pt")

with torch.no_grad():
  outputs = model(**inputs)
  logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
