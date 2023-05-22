# labels: test_group::monthly author::microsoft name::tapex-large-finetuned-tabfact downloads::201 license::mit task::Natural_Language_Processing sub_task::Text_Classification
from transformers import TapexTokenizer, BartForSequenceClassification
import pandas as pd

tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-tabfact")
model = BartForSequenceClassification.from_pretrained("microsoft/tapex-large-finetuned-tabfact")

data = {
    "year": [1896, 1900, 1904, 2004, 2008, 2012],
    "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]
}
table = pd.DataFrame.from_dict(data)

# tapex accepts uncased input since it is pre-trained on the uncased corpus
query = "beijing hosts the olympic games in 2012"
encoding = tokenizer(table=table, query=query, return_tensors="pt")

outputs = model(**encoding)
output_id = int(outputs.logits[0].argmax(dim=0))
print(model.config.id2label[output_id])
# Refused
