# labels: test_group::monthly author::tartuNLP name::EstBERT_NER downloads::884 license::cc-by-4.0 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline

tokenizer = BertTokenizer.from_pretrained('tartuNLP/EstBERT_NER')
bertner = BertForTokenClassification.from_pretrained('tartuNLP/EstBERT_NER')

nlp = pipeline("ner", model=bertner, tokenizer=tokenizer)
sentence = 'Eesti Ekspressi teada on Eesti Pank uurinud Hansapanga tehinguid , mis toimusid kaks aastat tagasi suvel ja mille käigus voolas panka ligi miljardi krooni ulatuses kahtlast raha .'

ner_results = nlp(sentence)
print(ner_results)
