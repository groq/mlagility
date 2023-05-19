# labels: test_group::monthly author::intfloat name::simlm-msmarco-reranker downloads::2,245 task::Natural_Language_Processing sub_task::Text_Classification
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput

def encode(tokenizer: PreTrainedTokenizerFast,
           query: str, passage: str, title: str = '-') -> BatchEncoding:
    return tokenizer(query,
                     text_pair='{}: {}'.format(title, passage),
                     max_length=192,
                     padding=True,
                     truncation=True,
                     return_tensors='pt')

tokenizer = AutoTokenizer.from_pretrained('intfloat/simlm-msmarco-reranker')
model = AutoModelForSequenceClassification.from_pretrained('intfloat/simlm-msmarco-reranker')
model.eval()

with torch.no_grad():
    batch_dict = encode(tokenizer, 'how long is super bowl game', 'The Super Bowl is typically four hours long. The game itself takes about three and a half hours, with a 30 minute halftime show built in.')
    outputs: SequenceClassifierOutput = model(**batch_dict, return_dict=True)
    print(outputs.logits[0])

    batch_dict = encode(tokenizer, 'how long is super bowl game', 'The cost of a Super Bowl commercial runs about $5 million for 30 seconds of airtime. But the benefits that the spot can bring to a brand can help to justify the cost.')
    outputs: SequenceClassifierOutput = model(**batch_dict, return_dict=True)
    print(outputs.logits[0])
