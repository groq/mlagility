# labels: test_group::monthly,daily author::facebook name::maskformer-swin-base-ade downloads::915 license::apache-2.0 task::Computer_Vision sub_task::Image_Segmentation
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-ade")
inputs = feature_extractor(images=image, return_tensors="pt")

model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")
outputs = model(**inputs)
# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to feature_extractor for postprocessing
output = feature_extractor.post_process_segmentation(outputs)
output = feature_extractor.post_process_semantic_segmentation(outputs)
output = feature_extractor.post_process_panoptic_segmentation(outputs)

