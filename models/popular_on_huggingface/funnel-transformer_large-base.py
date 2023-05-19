# labels: test_group::monthly author::funnel-transformer name::large-base downloads::264 license::apache-2.0 task::Multimodal sub_task::Feature_Extraction
from transformers import FunnelTokenizer, FunnelBaseModel
tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/large-base")
model = FunnelBaseModel.from_pretrained("funnel-transformer/large-base")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
