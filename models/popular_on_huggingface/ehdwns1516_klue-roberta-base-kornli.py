# labels: test_group::monthly author::ehdwns1516 name::klue-roberta-base-kornli downloads::390 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("ehdwns1516/klue-roberta-base-kornli")

classifier = pipeline(
    "text-classification",
    model="ehdwns1516/klue-roberta-base-kornli",
    return_all_scores=True,
)

premise = "your premise"
hypothesis = "your hypothesis"

result = dict()
result[0] = classifier(premise + tokenizer.sep_token + hypothesis)[0]
