# labels: test_group::monthly author::avichr name::heBERT_sentiment_analysis downloads::2,393 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
model = AutoModel.from_pretrained("avichr/heBERT")
    
from transformers import pipeline
fill_mask = pipeline(
    "fill-mask",
    model="avichr/heBERT",
    tokenizer="avichr/heBERT"
)
fill_mask("הקורונה לקחה את [MASK] ולנו לא נשאר דבר.")
